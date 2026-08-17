// Headless JWildfire render of a .flame file to PNG, for the flame comparison
// harness (src/dev/flameCompare.ts + scripts/jwf-port/compare.py).
//
//   java -Djava.awt.headless=true -cp "tools/out:$CP" RenderOne in.flame out.png W H quality
//
// The flame is rendered at W×H with its own pixels-per-unit scaled by W/flame.width
// (like a JWildfire "render at size" — same framing), sample density = quality,
// spatial oversampling 1, no post-processing beyond the flame's own settings.
import java.io.File;
import java.util.List;
import javax.imageio.ImageIO;
import org.jwildfire.base.Prefs;
import org.jwildfire.base.Tools;
import org.jwildfire.create.tina.base.Flame;
import org.jwildfire.create.tina.io.FlameReader;
import org.jwildfire.create.tina.render.FlameRenderer;
import org.jwildfire.create.tina.render.RenderInfo;
import org.jwildfire.create.tina.render.RenderMode;
import org.jwildfire.create.tina.render.RenderedFlame;

public class RenderOne {
  public static void main(String[] args) throws Exception {
    String in = args[0], out = args[1];
    int w = Integer.parseInt(args[2]), h = Integer.parseInt(args[3]);
    double quality = Double.parseDouble(args[4]);
    Prefs prefs = Prefs.getPrefs();
    String xml = new String(Tools.readFile(in), "UTF-8");
    List<Flame> flames = new FlameReader(prefs).readFlamesfromXML(xml);
    Flame flame = flames.get(0);
    double sx = (double) w / (double) flame.getWidth();
    flame.setWidth(w);
    flame.setHeight(h);
    flame.setPixelsPerUnit(flame.getPixelsPerUnit() * sx);
    flame.setSampleDensity(quality);
    flame.setSpatialOversampling(1);
    flame.setBGTransparency(false);
    FlameRenderer renderer = new FlameRenderer(flame, prefs, false, false);
    RenderInfo info = new RenderInfo(w, h, RenderMode.PRODUCTION);
    RenderedFlame res = renderer.renderFlame(info);
    ImageIO.write(res.getImage().getBufferedImg(), "png", new File(out));
    System.out.println("ok " + out);
  }
}
