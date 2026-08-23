import java.awt.Color;
public class CProbe {
  public static void main(String[] a) {
    int[] ns = {1,2,3,5,7,10,50,100,1394,2390,20000,20001};
    StringBuilder sb = new StringBuilder("[");
    for (int n : ns) {
      float z = (float) Math.log(n); float h = z - (float) Math.floor(z);
      Color c1 = n == 20001 ? Color.black : Color.getHSBColor(h, (float) 0.85, 1);
      float[] hsb = Color.RGBtoHSB(c1.getRed(), c1.getGreen(), c1.getBlue(), new float[3]);
      float b = (float) (n % 17) / 16, r = (float) (n % 91) / 90, gr = (float) (n % 123) / 122;
      Color c0 = new Color(r, gr, b);
      float[] hsb0 = Color.RGBtoHSB(c0.getRed(), c0.getGreen(), c0.getBlue(), new float[3]);
      float z2 = (float) (n / 6.333333); float h2 = z2 - (float) Math.floor(z2);
      Color c2 = Color.getHSBColor(h2, (float) 0.85, 1);
      sb.append("[" + n + ",[" + c1.getRed() + "," + c1.getGreen() + "," + c1.getBlue() + "]," + hsb[0] + ",[" + c0.getRed() + "," + c0.getGreen() + "," + c0.getBlue() + "]," + hsb0[0] + ",[" + c2.getRed() + "," + c2.getGreen() + "," + c2.getBlue() + "]],");
    }
    sb.setLength(sb.length() - 1); sb.append("]");
    System.out.println(sb);
  }
}
