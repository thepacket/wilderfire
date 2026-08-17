import org.jwildfire.create.tina.variation.*;
import org.jwildfire.create.tina.base.*;
import org.jwildfire.create.tina.random.*;

import java.io.*;
import java.util.*;

/** Dumps every JWildfire variation's metadata + GPU code as JSON lines. */
public class Dump {
  static String q(String s) {
    if (s == null) return "null";
    StringBuilder b = new StringBuilder("\"");
    for (char c : s.toCharArray()) {
      switch (c) {
        case '"': b.append("\\\""); break;
        case '\\': b.append("\\\\"); break;
        case '\n': b.append("\\n"); break;
        case '\r': b.append("\\r"); break;
        case '\t': b.append("\\t"); break;
        default:
          if (c < 0x20) b.append(String.format("\\u%04x", (int) c)); else b.append(c);
      }
    }
    return b.append('"').toString();
  }

  static FlameTransformationContext ctx() {
    AbstractRandomGenerator rg = new MarsagliaRandomGenerator();
    rg.randomize(12345);
    FlameTransformationContext c = new FlameTransformationContext(null, rg, 0, 0);
    c.setPreserveZCoordinate(false);
    return c;
  }

  static String gpuCode(VariationFunc f, boolean perturb) {
    try {
      VariationFunc copy = f.makeCopy();
      if (perturb) {
        String[] names = copy.getParameterNames();
        Object[] vals = copy.getParameterValues();
        for (int i = 0; i < names.length; i++) {
          double v = ((Number) vals[i]).doubleValue();
          try { copy.setParameter(names[i], (vals[i] instanceof Integer) ? v + 1 : v + 0.37); } catch (Throwable t) { }
        }
      }
      FlameTransformationContext c = ctx();
      copy.init(c, new Layer(), new XForm(), 1.0);
      return ((SupportsGPU) copy).getGPUCode(c);
    } catch (Throwable t) {
      return "/*ERROR " + t.getClass().getSimpleName() + ": " + t.getMessage() + "*/";
    }
  }

  static String gpuFuncs(VariationFunc f) {
    try {
      VariationFunc copy = f.makeCopy();
      FlameTransformationContext c = ctx();
      copy.init(c, new Layer(), new XForm(), 1.0);
      return ((SupportsGPU) copy).getGPUFunctions(c);
    } catch (Throwable t) {
      return "/*ERROR " + t.getClass().getSimpleName() + ": " + t.getMessage() + "*/";
    }
  }

  public static void main(String[] args) throws Exception {
    List<String> names = VariationFuncList.getNameList();
    PrintStream out = new PrintStream(new FileOutputStream(args[0]), true, "UTF-8");
    int n = 0;
    for (String name : names) {
      VariationFunc f, f2;
      try {
        f = VariationFuncList.getVariationFuncInstance(name);
        f2 = VariationFuncList.getVariationFuncInstance(name);
      } catch (Throwable t) {
        out.println("{\"name\":" + q(name) + ",\"error\":" + q(t.toString()) + "}");
        continue;
      }
      StringBuilder sb = new StringBuilder("{");
      sb.append("\"name\":").append(q(name));
      sb.append(",\"cls\":").append(q(f.getClass().getSimpleName()));
      sb.append(",\"priority\":").append(f.getPriority());
      sb.append(",\"types\":[");
      VariationFuncType[] ts = f.getVariationTypes();
      for (int i = 0; ts != null && i < ts.length; i++) { if (i > 0) sb.append(','); sb.append(q(ts[i].name())); }
      sb.append("]");
      // params
      sb.append(",\"params\":[");
      String[] pn = f.getParameterNames();
      Object[] pv = f.getParameterValues();
      Object[] pv2 = f2.getParameterValues();
      boolean stable = true;
      for (int i = 0; pn != null && i < pn.length; i++) {
        if (i > 0) sb.append(',');
        Object v = pv[i];
        boolean isInt = v instanceof Integer;
        double d = v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
        double d2 = pv2[i] instanceof Number ? ((Number) pv2[i]).doubleValue() : Double.NaN;
        boolean same = (Double.isNaN(d) && Double.isNaN(d2)) || d == d2;
        if (!same) stable = false;
        sb.append("{\"name\":").append(q(pn[i])).append(",\"def\":").append(Double.isNaN(d) ? "null" : String.valueOf(d))
          .append(",\"int\":").append(isInt).append(",\"stable\":").append(same).append("}");
      }
      sb.append("]");
      String[] alt = f.getParameterAlternativeNames();
      if (alt != null) {
        sb.append(",\"altNames\":[");
        for (int i = 0; i < alt.length; i++) { if (i > 0) sb.append(','); sb.append(q(alt[i])); }
        sb.append("]");
      }
      // resources = things the variation reads at run time (images, code, text); a REFERENCE
      // ressource is only a link to the author's page and does not make it unportable
      String[] rn = f.getRessourceNames();
      int nres = 0;
      if (rn != null) for (String r : rn) { RessourceType rt = null; try { rt = f.getRessourceType(r); } catch (Throwable t) { } if (rt != RessourceType.REFERENCE) nres++; }
      sb.append(",\"resources\":").append(nres);
      sb.append(",\"defaultsStable\":").append(stable);
      boolean gpu = f instanceof SupportsGPU;
      sb.append(",\"gpu\":").append(gpu);
      if (gpu) {
        String c1 = gpuCode(f, false);
        String c2 = gpuCode(f, true);
        sb.append(",\"gpuCode\":").append(q(c1));
        sb.append(",\"gpuCodeParamDependent\":").append(!Objects.equals(c1, c2));
        if (!Objects.equals(c1, c2)) sb.append(",\"gpuCodePerturbed\":").append(q(c2));
        sb.append(",\"gpuFunctions\":").append(q(gpuFuncs(f)));
        sb.append(",\"stateful\":").append(((SupportsGPU) f).isStateful());
        String[] ex = ((SupportsGPU) f).getGPUExtraParameterNames();
        sb.append(",\"extraParams\":[");
        for (int i = 0; ex != null && i < ex.length; i++) { if (i > 0) sb.append(','); sb.append(q(ex[i])); }
        sb.append("]");
      }
      sb.append("}");
      out.println(sb);
      n++;
    }
    out.close();
    System.err.println("dumped " + n + " variations");
  }
}
