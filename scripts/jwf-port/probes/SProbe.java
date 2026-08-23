import org.jwildfire.create.tina.variation.SnowflakeWFFunc;
import java.lang.reflect.*;
import java.util.*;
public class SProbe {
  public static void main(String[] a) throws Exception {
    SnowflakeWFFunc f = new SnowflakeWFFunc();
    String[] names = {"buffer_size","max_iter","bg_freeze_level","fg_freeze_speed","diffusion_speed","diffusion_asymmetry","rnd_bg_noise","threshold","seed","scale","jitter","dc_color","dc_color_scale","dc_color_offset"};
    double[] vals = {128,500,0.5,0.0005,0.01,1.0,0.25,0.65,12345,1.0,0.001,1,2.0,0.1};
    if (a.length > 0) { vals[0] = Double.parseDouble(a[0]); vals[1] = Double.parseDouble(a[1]); }
    for (int i = 0; i < names.length; i++) f.setParameter(names[i], vals[i]);
    f.init(null, null, null, 1.0);
    Field pf = SnowflakeWFFunc.class.getDeclaredField("_points"); pf.setAccessible(true);
    List<?> pts = (List<?>) pf.get(f);
    StringBuilder sb = new StringBuilder("[");
    for (Object p : pts) {
      Class<?> c = p.getClass();
      Field fx = c.getDeclaredField("x"), fy = c.getDeclaredField("y"), fi = c.getDeclaredField("intensity");
      fx.setAccessible(true); fy.setAccessible(true); fi.setAccessible(true);
      sb.append('[').append(fx.getDouble(p)).append(',').append(fy.getDouble(p)).append(',').append(fi.getFloat(p)).append("],");
    }
    if (pts.size() > 0) sb.setLength(sb.length() - 1);
    sb.append(']');
    System.out.println(sb);
  }
}
