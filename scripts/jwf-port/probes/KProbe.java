import org.jwildfire.create.tina.variation.KleinGroupFunc;
import org.nfunk.jep.type.Complex;
import java.lang.reflect.Field;
public class KProbe {
  public static void main(String[] a) throws Exception {
    double[][] sets = { {0,2,0,2,0},{1,2,1,2,1},{2,2,0,1,1},{3,1,2.056,0.584,0},{4,1,2.056,0.584,0},{5,2,1,2,1},{6,2,1,2,1},{0,2,1,2,1},{2,1.7,0.3,2.2,-0.5},{6,1.5,0.7,0.4,1.1} };
    for (double[] s : sets) {
      KleinGroupFunc f = new KleinGroupFunc();
      f.setParameter("recipe", s[0]); f.setParameter("a_re", s[1]); f.setParameter("a_im", s[2]); f.setParameter("b_re", s[3]); f.setParameter("b_im", s[4]);
      f.init(null, null, null, 1.0);
      StringBuilder sb = new StringBuilder("{\"recipe\":" + (int)s[0] + ",\"a_re\":" + s[1] + ",\"a_im\":" + s[2] + ",\"b_re\":" + s[3] + ",\"b_im\":" + s[4] + ",\"mats\":[");
      String[] names = {"mat_a","mat_inv_a","mat_b","mat_inv_b"};
      for (int i = 0; i < 4; i++) {
        Field fl = KleinGroupFunc.class.getDeclaredField(names[i]); fl.setAccessible(true);
        Complex[] m = (Complex[]) fl.get(f);
        for (int k = 0; k < 4; k++) sb.append(m[k].re()).append(',').append(m[k].im()).append(',');
      }
      sb.setLength(sb.length()-1); sb.append("]}");
      System.out.println(sb);
    }
  }
}
