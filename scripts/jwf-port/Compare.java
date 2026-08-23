// Numeric comparison of WilderFire vs headless JWildfire renders (no Python deps).
//
//   1. dev browser: await window.wilderfire.flameCompare()   → compare-out/<id>.wf.png + <id>.flame + manifest.json
//   2. cd <jwf>; java -Djava.awt.headless=true -cp "tools/out:$CP" Compare <repo>/compare-out [ids…]
//
// Renders each <id>.flame with JWildfire at the manifest's size/quality (cached as
// <id>.jwf.png), then prints one line per flame: mean luma of both, ratio, coverage
// (fraction of non-background pixels), 16×16-block MAE (0-255), luma-histogram,
// per-channel mean ratios R/G/B (flag `channel` when a channel's ratio strays > 0.2 from the luma ratio)
// intersection and a downscaled-greyscale correlation, plus flags. Nothing is judged
// visually.
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.regex.*;
import javax.imageio.ImageIO;
import org.jwildfire.base.Prefs;
import org.jwildfire.base.Tools;
import org.jwildfire.create.tina.base.Flame;
import org.jwildfire.create.tina.io.FlameReader;
import org.jwildfire.create.tina.render.*;

public class Compare {
  static BufferedImage renderJwf(File flameFile, int w, int h, double quality) throws Exception {
    Prefs prefs = Prefs.getPrefs();
    String xml = new String(Tools.readFile(flameFile.getPath()), "UTF-8");
    Flame flame = new FlameReader(prefs).readFlamesfromXML(xml).get(0);
    double sx = (double) w / (double) flame.getWidth();
    flame.setWidth(w); flame.setHeight(h);
    flame.setPixelsPerUnit(flame.getPixelsPerUnit() * sx);
    flame.setSampleDensity(quality);
    flame.setSpatialOversampling(1);
    flame.setBGTransparency(false);
    FlameRenderer renderer = new FlameRenderer(flame, prefs, false, false);
    RenderedFlame res = renderer.renderFlame(new RenderInfo(w, h, RenderMode.PRODUCTION));
    return res.getImage().getBufferedImg();
  }
  static double[][] luma(BufferedImage img) {
    int w = img.getWidth(), h = img.getHeight();
    double[][] l = new double[h][w];
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
      int p = img.getRGB(x, y);
      l[y][x] = 0.299 * ((p >> 16) & 255) + 0.587 * ((p >> 8) & 255) + 0.114 * (p & 255);
    }
    return l;
  }
  /** per-channel means (R, G, B) — a luma ratio can hide a colour-scale error cancelling a pattern error */
  static double[] channels(BufferedImage img) {
    int w = img.getWidth(), h = img.getHeight(); double r = 0, g = 0, b = 0;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { int p = img.getRGB(x, y); r += (p >> 16) & 255; g += (p >> 8) & 255; b += p & 255; }
    double n = (double) w * h; return new double[] { r / n, g / n, b / n };
  }
  static double mean(double[][] l) { double s = 0; for (double[] r : l) for (double v : r) s += v; return s / (l.length * l[0].length); }
  static double cover(double[][] l) { double s = 0; for (double[] r : l) for (double v : r) if (v > 8) s++; return s / (l.length * l[0].length); }
  static double[][] blocks(double[][] l, int n) {
    int h = l.length, w = l[0].length, bh = h / n, bw = w / n;
    double[][] b = new double[n][n];
    for (int by = 0; by < n; by++) for (int bx = 0; bx < n; bx++) {
      double s = 0; for (int y = by * bh; y < (by + 1) * bh; y++) for (int x = bx * bw; x < (bx + 1) * bw; x++) s += l[y][x];
      b[by][bx] = s / (bh * bw);
    }
    return b;
  }
  public static void main(String[] args) throws Exception {
    File out = new File(args[0]);
    Set<String> only = new HashSet<>(Arrays.asList(args).subList(1, args.length));
    String manName = System.getenv("MANIFEST") != null ? System.getenv("MANIFEST") : "manifest.json";
    String man = new String(Tools.readFile(new File(out, manName).getPath()), "UTF-8");
    Matcher m = Pattern.compile("\\{[^{}]*\"id\":\\s*\"([^\"]+)\"[^{}]*\"w\":\\s*(\\d+)[^{}]*\"h\":\\s*(\\d+)[^{}]*\"quality\":\\s*([\\d.]+)[^{}]*\\}").matcher(man);
    System.out.printf("%-28s %7s %7s %6s %6s %6s %7s %5s %5s  %-14s flags%n", "flame", "lumaWF", "lumaJW", "ratio", "covWF", "covJW", "blkMAE", "hist", "corr", "rgbRatio");
    int n = 0; List<String> flagged = new ArrayList<>();
    while (m.find()) {
      String id = m.group(1); int w = Integer.parseInt(m.group(2)), h = Integer.parseInt(m.group(3)); double q = Double.parseDouble(m.group(4));
      if (!only.isEmpty() && !only.contains(id)) continue;
      try {
        File jf = new File(out, id + ".jwf.png"), ff = new File(out, id + ".flame"), wf = new File(out, id + ".wf.png");
        BufferedImage jimg;
        if (jf.exists() && jf.lastModified() >= ff.lastModified()) jimg = ImageIO.read(jf);
        else { jimg = renderJwf(ff, w, h, q); ImageIO.write(jimg, "png", jf); }
        BufferedImage wimg = ImageIO.read(wf);
        double[][] la = luma(wimg), lb = luma(jimg);
        double ma = mean(la), mb = mean(lb), ratio = (ma + 1e-6) / (mb + 1e-6);
        double ca = cover(la), cb = cover(lb);
        double[][] ba = blocks(la, 16), bb = blocks(lb, 16);
        double mae = 0; for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) mae += Math.abs(ba[i][j] - bb[i][j]); mae /= 256;
        double[] ha = new double[64], hb = new double[64];
        for (double[] r : la) for (double v : r) ha[Math.min(63, (int) (v / 4))]++;
        for (double[] r : lb) for (double v : r) hb[Math.min(63, (int) (v / 4))]++;
        double hist = 0, na = la.length * la[0].length, nb = lb.length * lb[0].length; for (int i = 0; i < 64; i++) hist += Math.min(ha[i] / na, hb[i] / nb);
        // structure: correlation of 32×32 block means
        double[][] sa = blocks(la, 32), sb = blocks(lb, 32);
        double mA = 0, mB = 0; for (int i = 0; i < 32; i++) for (int j = 0; j < 32; j++) { mA += sa[i][j]; mB += sb[i][j]; } mA /= 1024; mB /= 1024;
        double sab = 0, saa = 0, sbb = 0; for (int i = 0; i < 32; i++) for (int j = 0; j < 32; j++) { double da = sa[i][j] - mA, db = sb[i][j] - mB; sab += da * db; saa += da * da; sbb += db * db; }
        double corr = sab / (Math.sqrt(saa * sbb) + 1e-9);
        List<String> flags = new ArrayList<>();
        if (ratio < 0.7 || ratio > 1.4) flags.add("brightness");
        if (Math.abs(ca - cb) > 0.15) flags.add("coverage");
        if (mae > 25) flags.add("blocks");
        double[] cA = channels(wimg), cB = channels(jimg);
        String rgbRatio = String.format("%.2f/%.2f/%.2f", (cA[0] + 1e-6) / (cB[0] + 1e-6), (cA[1] + 1e-6) / (cB[1] + 1e-6), (cA[2] + 1e-6) / (cB[2] + 1e-6));
        if (corr < 0.8) flags.add("structure");
        for (int c = 0; c < 3; c++) { double rc = (cA[c] + 1e-6) / (cB[c] + 1e-6); if (cB[c] > 2 && Math.abs(rc - ratio) > 0.2) { flags.add("channel"); break; } }
        if (!flags.isEmpty()) flagged.add(id);
        n++;
        System.out.printf("%-28s %7.1f %7.1f %6.2f %6.2f %6.2f %7.1f %5.2f %5.2f  %-14s %s%n", id, ma, mb, ratio, ca, cb, mae, hist, corr, rgbRatio, String.join(" ", flags));
      } catch (Throwable e) {
        System.out.printf("%-28s ERROR %s%n", id, String.valueOf(e).replace('\n', ' '));
      }
    }
    System.out.printf("%n%d compared, %d flagged: %s%n", n, flagged.size(), String.join(" ", flagged));
  }
}
