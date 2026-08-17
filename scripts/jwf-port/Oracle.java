import org.jwildfire.create.tina.variation.*;
import org.jwildfire.create.tina.base.*;
import org.jwildfire.create.tina.random.*;

import java.io.*;
import java.util.*;

/**
 * Headless JWildfire oracle: evaluates each variation's Java transform() on the
 * points/param sets from oracle-spec.txt and writes JSON lines to stdout-file.
 * Deterministic variations get one sample per point; variations that consume
 * randomness get SAMPLES samples per point (raw values, stats computed by the
 * browser harness).
 *
 *   java -cp "out:lib/*:tools/out" Oracle oracle-spec.txt oracle-out.jsonl
 */
public class Oracle {
  static final int SAMPLES = 256;

  /** RNG wrapper that counts how many random numbers a transform consumed. */
  // Mersenne Twister rather than JWildfire's Marsaglia MWC: Marsaglia's randomize()
  // shifts the seed by 16 bits and some seeds start it in a degenerate state (e.g.
  // solidangle3D's name hash never sampled a point inside its solid).
  static class CountingRandom extends AbstractRandomGenerator {
    final MersenneTwisterRandomGenerator inner = new MersenneTwisterRandomGenerator();
    long count = 0;
    CountingRandom(long seed) { inner.randomize(seed); }
    @Override public double random() { count++; return inner.random(); }
    @Override public int random(int pMax) { count++; return inner.random(pMax); }
    @Override public void randomize(long pSeed) { inner.randomize(pSeed); }
    @Override public void cleanup() { }
  }

  static String num(double d) {
    if (Double.isNaN(d)) return "null";
    if (Double.isInfinite(d) || Math.abs(d) > 1e30) return d > 0 ? "1e30" : "-1e30";
    return String.valueOf((float) d);
  }

  public static void main(String[] args) throws Exception {
    BufferedReader in = new BufferedReader(new FileReader(args[0]));
    PrintStream out = new PrintStream(new FileOutputStream(args[1]), true, "UTF-8");
    String line = in.readLine();
    int np = Integer.parseInt(line.split(" ")[1]);
    double[] px = new double[np], py = new double[np], pz = new double[np];
    for (int i = 0; i < np; i++) {
      String[] t = in.readLine().trim().split(" ");
      px[i] = Double.parseDouble(t[0]); py[i] = Double.parseDouble(t[1]); pz[i] = t.length > 2 ? Double.parseDouble(t[2]) : 0;
    }
    String[] af = in.readLine().split(" ");
    double[] affine = new double[6];
    for (int i = 0; i < 6; i++) affine[i] = Double.parseDouble(af[i + 1]);

    String curName = null; int curPrio = 0; int setIdx = 0;
    int nVars = 0;
    while ((line = in.readLine()) != null) {
      line = line.trim();
      if (line.startsWith("VAR ")) {
        String[] t = line.split(" ");
        curName = t[1]; curPrio = Integer.parseInt(t[2]); setIdx = 0; nVars++;
        continue;
      }
      if (!line.startsWith("SET ")) continue;
      String[] t = line.split(" ");
      double weight = Double.parseDouble(t[1]);
      int nParams = Integer.parseInt(t[2]);
      LinkedHashMap<String, Double> params = new LinkedHashMap<>();
      for (int i = 0; i < nParams; i++) {
        String[] kv = in.readLine().trim().split(" ");
        params.put(kv[0], Double.parseDouble(kv[1]));
      }
      final int si = setIdx++;
      StringBuilder sb = new StringBuilder();
      sb.append("{\"name\":\"").append(curName).append("\",\"set\":").append(si);
      try {
        VariationFunc f = VariationFuncList.getVariationFuncInstance(curName, true);
        for (Map.Entry<String, Double> e : params.entrySet()) {
          try { f.setParameter(e.getKey(), e.getValue()); } catch (Throwable ex) { sb.append(",\"paramError\":\"").append(e.getKey()).append("\""); }
        }
        CountingRandom rng = new CountingRandom(1234567L + curName.hashCode());
        FlameTransformationContext ctx = new FlameTransformationContext(null, rng, 0, 0);
        ctx.setPreserveZCoordinate(false);
        XForm xf = new XForm();
        xf.setXYCoeff00(affine[0]); xf.setXYCoeff10(affine[1]); xf.setXYCoeff20(affine[2]);
        xf.setXYCoeff01(affine[3]); xf.setXYCoeff11(affine[4]); xf.setXYCoeff21(affine[5]);
        Layer layer = new Layer();
        f.initOnce(ctx, layer, xf, weight);
        f.init(ctx, layer, xf, weight);
        // Detect randomness with a dry run over all points: count context RNG
        // calls, and also evaluate twice (some variations call Math.random()).
        XYZPoint a = new XYZPoint(), v = new XYZPoint();
        boolean differs = false;
        for (int i = 0; i < np; i++) {
          double x1 = Double.NaN, y1 = Double.NaN, x2 = Double.NaN, y2 = Double.NaN;
          setup(a, v, px[i], py[i], pz[i], curPrio);
          try { f.transform(ctx, xf, a, v, weight); XYZPoint o = curPrio == -1 ? a : v; x1 = o.x; y1 = o.y; } catch (Throwable ex) { }
          setup(a, v, px[i], py[i], pz[i], curPrio);
          try { f.transform(ctx, xf, a, v, weight); XYZPoint o = curPrio == -1 ? a : v; x2 = o.x; y2 = o.y; } catch (Throwable ex) { }
          if ((x1 != x2 && !(Double.isNaN(x1) && Double.isNaN(x2))) || (y1 != y2 && !(Double.isNaN(y1) && Double.isNaN(y2)))) differs = true;
        }
        boolean random = rng.count > 0 || differs;
        int samples = random ? SAMPLES : 1;
        // Re-init for a clean state (some variations keep state)
        f = VariationFuncList.getVariationFuncInstance(curName, true);
        for (Map.Entry<String, Double> e : params.entrySet()) { try { f.setParameter(e.getKey(), e.getValue()); } catch (Throwable ex) { } }
        f.initOnce(ctx, layer, xf, weight);
        f.init(ctx, layer, xf, weight);
        sb.append(",\"random\":").append(random).append(",\"samples\":").append(samples).append(",\"out\":[");
        boolean first = true;
        String err = null;
        for (int i = 0; i < np; i++) {
          // accumulate stats over samples (samples==1 → plain values)
          double sx = 0, sy = 0, sz = 0, sxx = 0, syy = 0, szz = 0, sc = 0; int hides = 0, valid = 0;
          for (int s = 0; s < samples; s++) {
            setup(a, v, px[i], py[i], pz[i], curPrio);
            double ox, oy, oz, oc; boolean hide;
            try {
              f.transform(ctx, xf, a, v, weight);
              XYZPoint o = curPrio == -1 ? a : v;
              ox = o.x; oy = o.y; oz = o.z; oc = v.color; hide = v.doHide || a.doHide;
            } catch (Throwable ex) {
              ox = Double.NaN; oy = Double.NaN; oz = Double.NaN; oc = Double.NaN; hide = false;
              if (err == null) err = ex.getClass().getSimpleName() + ": " + ex.getMessage();
            }
            if (Double.isNaN(ox) || Double.isNaN(oy) || Double.isNaN(oz) || Double.isInfinite(ox) || Double.isInfinite(oy) || Double.isInfinite(oz)) continue;
            valid++;
            sx += ox; sy += oy; sz += oz; sxx += ox * ox; syy += oy * oy; szz += oz * oz; sc += oc; if (hide) hides++;
          }
          if (!first) sb.append(',');
          first = false;
          if (valid == 0) { sb.append("null"); continue; }
          double mx = sx / valid, my = sy / valid, mz = sz / valid;
          double vx = Math.max(0, sxx / valid - mx * mx), vy = Math.max(0, syy / valid - my * my), vz = Math.max(0, szz / valid - mz * mz);
          sb.append('[').append(num(mx)).append(',').append(num(my)).append(',').append(num(sc / valid)).append(',').append(num((double) hides / valid));
          if (samples > 1) sb.append(',').append(num(Math.sqrt(vx))).append(',').append(num(Math.sqrt(vy))).append(',').append(num(mz)).append(',').append(num(Math.sqrt(vz)));
          else sb.append(',').append(num(mz));
          sb.append(']');
        }
        sb.append("]");
        if (err != null) sb.append(",\"error\":\"").append(err.replace("\"", "'").replace("\\", "/")).append("\"");
      } catch (Throwable ex) {
        sb.append(",\"error\":\"").append((ex.getClass().getSimpleName() + ": " + ex.getMessage()).replace("\"", "'").replace("\\", "/")).append("\"");
      }
      sb.append("}");
      out.println(sb);
    }
    out.close();
    System.err.println("oracle: " + nVars + " variations");
  }

  static void setup(XYZPoint a, XYZPoint v, double x, double y, double z, int prio) {
    a.clear(); v.clear();
    a.x = x; a.y = y; a.z = z; a.color = 0.5;
    a.invalidate();
    v.color = 0.5;
    if (prio == 1) { v.x = x; v.y = y; v.z = z; }
    else { v.x = 0; v.y = 0; v.z = 0; }
    v.invalidate();
  }
}
