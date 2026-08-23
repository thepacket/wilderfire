import org.jwildfire.create.tina.variation.*;
import java.lang.reflect.*;
import java.util.*;
public class MProbe {
  static Object get(Object o, String f) throws Exception { Class<?> c = o.getClass(); while (c != null) { try { Field x = c.getDeclaredField(f); x.setAccessible(true); return x.get(o); } catch (NoSuchFieldException e) { c = c.getSuperclass(); } } throw new NoSuchFieldException(f); }
  public static void main(String[] a) throws Exception {
    String which = a[0];
    StringBuilder sb = new StringBuilder();
    if (which.equals("mandala")) {
      MandalaFunc f = new MandalaFunc();
      String[] n = {"width","size","num","denom","minsky","wobble","wrap_range","hskew","color id"};
      for (int i = 0; i < n.length; i++) f.setParameter(n[i], Double.parseDouble(a[i + 1]));
      f.init(null, null, null, 1.0);
      List<?> prims = (List<?>) get(f, "primitives");
      sb.append("{\"count\":" + prims.size() + ",\"items\":[");
      int k = 0;
      for (Object p : prims) {
        Object pos = get(p, "pos");
        double px = ((Number) get(pos, "x")).doubleValue(), py = ((Number) get(pos, "y")).doubleValue();
        double col = ((Number) get(p, "color")).doubleValue();
        Object rgb = null; try { rgb = get(p, "rgbColor"); } catch (Exception e) { try { rgb = get(p, "rgbcolor"); } catch (Exception e2) {} }
        String rs = rgb == null ? "null" : ("[" + ((java.awt.Color) rgb).getRed() + "," + ((java.awt.Color) rgb).getGreen() + "," + ((java.awt.Color) rgb).getBlue() + "]");
        if (k < 20 || k % 997 == 0) sb.append("[" + k + "," + px + "," + py + "," + col + "," + rs + "],");
        k++;
      }
      if (sb.charAt(sb.length() - 1) == ',') sb.setLength(sb.length() - 1);
      sb.append("]}");
      // field names of the first primitive
      if (prims.size() > 0) { Class<?> c = prims.get(0).getClass(); StringBuilder fn = new StringBuilder(); while (c != null) { for (Field x : c.getDeclaredFields()) fn.append(x.getName() + ":" + x.getType().getSimpleName() + " "); c = c.getSuperclass(); } System.err.println("FIELDS " + prims.get(0).getClass().getName() + " " + fn); }
    } else {
      Mandala2Func f = new Mandala2Func();
      String[] n = {"width","num","denom","minsky","wobble","wrap_range","hskew","color id"};
      for (int i = 0; i < n.length; i++) f.setParameter(n[i], Double.parseDouble(a[i + 1]));
      f.initOnce(null, null, null, 1.0);
      int[][] sc = (int[][]) get(f, "step_counts");
      sb.append("{\"w\":" + sc.length + ",\"h\":" + sc[0].length + ",\"cells\":[");
      long sum = 0; int mx = 0;
      for (int x = 0; x < sc.length; x++) for (int y = 0; y < sc[x].length; y++) { sum += sc[x][y]; mx = Math.max(mx, sc[x][y]); }
      for (int x = 0; x < sc.length; x += 37) for (int y = 0; y < sc[x].length; y += 41) sb.append("[" + x + "," + y + "," + sc[x][y] + "],");
      sb.setLength(sb.length() - 1);
      sb.append("],\"sum\":" + sum + ",\"max\":" + mx + "}");
    }
    System.out.println(sb);
  }
}
