import org.jwildfire.create.tina.variation.*;
import java.lang.reflect.*;
import java.util.*;
public class VProbe {
  static Object get(Object o, String f) throws Exception { Class<?> c = o.getClass(); while (c != null) { try { Field x = c.getDeclaredField(f); x.setAccessible(true); return x.get(o); } catch (NoSuchFieldException e) { c = c.getSuperclass(); } } throw new NoSuchFieldException(f); }
  public static void main(String[] a) throws Exception {
    SunflowerVoroniFunc f = new SunflowerVoroniFunc();
    String[] n = {"nPoints","Iters","angle","color mode","outline","fill","outline color"};
    for (int i = 0; i < n.length; i++) f.setParameter(n[i], Double.parseDouble(a[i]));
    f.init(null, null, null, 1.0);
    float[][] pts = (float[][]) get(f, "points");
    Object vor = get(f, "voroni");
    float[][] edges = (float[][]) get(vor, "edges");
    Object[] regions = (Object[]) get(vor, "regions");
    StringBuilder sb = new StringBuilder("{\"points\":[");
    for (float[] p : pts) sb.append("[" + p[0] + "," + p[1] + "],");
    sb.setLength(sb.length() - 1); sb.append("],\"edges\":[");
    for (float[] e : edges) sb.append("[" + e[0] + "," + e[1] + "," + e[2] + "," + e[3] + "],");
    if (edges.length > 0) sb.setLength(sb.length() - 1); sb.append("],\"regions\":[");
    for (Object r : regions) { float[][] c = (float[][]) get(r, "coords"); int cnt = (Integer) get(r, "count"); sb.append("["); for (int i = 0; i < cnt; i++) sb.append("[" + c[i][0] + "," + c[i][1] + "],"); if (cnt > 0) sb.setLength(sb.length() - 1); sb.append("],"); }
    sb.setLength(sb.length() - 1); sb.append("],\"prims\":[");
    List<?> prims = (List<?>) get(f, "primitives");
    for (Object p : prims) {
      int type = (Integer) get(p, "type");
      if (type == 2) { sb.append("[2," + get(p, "x") + "," + get(p, "y") + "," + get(p, "x2") + "," + get(p, "y2") + "," + get(p, "color") + "],"); }
      else if (type == 3) { Object p1 = get(p, "p1"), p2 = get(p, "p2"), p3 = get(p, "p3"); sb.append("[3," + get(p1, "x") + "," + get(p1, "y") + "," + get(p2, "x") + "," + get(p2, "y") + "," + get(p3, "x") + "," + get(p3, "y") + "," + get(p, "color") + "],"); }
      else sb.append("[" + type + "],");
    }
    if (prims.size() > 0) sb.setLength(sb.length() - 1); sb.append("]}");
    System.out.println(sb);
    if (prims.size() > 0) { Class<?> c = prims.get(0).getClass(); StringBuilder fn = new StringBuilder(); while (c != null) { for (Field x : c.getDeclaredFields()) fn.append(x.getName() + ":" + x.getType().getSimpleName() + " "); c = c.getSuperclass(); } System.err.println("FIELDS " + prims.get(0).getClass().getName() + " " + fn); }
  }
}
