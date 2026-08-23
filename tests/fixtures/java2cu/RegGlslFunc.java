package org.jwildfire.create.tina.variation;

import js.glsl.G;
import js.glsl.vec2;
import org.jwildfire.base.Tools;
import org.jwildfire.create.tina.base.XForm;
import org.jwildfire.create.tina.base.XYZPoint;

/** Synthetic regression fixture: js.glsl-style code — G.atan2 (JWildfire's rational approximation, not
 *  Math.atan2), the two-argument G.atan(n, d) = atan(n/d), G.Kscope (which uses G.atan2 inside), vec2 chains. */
public class RegGlslFunc extends VariationFunc {
  private static final long serialVersionUID = 1L;
  private static final String PARAM_ZOOM = "zoom";
  private static final String PARAM_INVERT = "invert";
  private static final String[] additionalParamNames = {PARAM_ZOOM, PARAM_INVERT};

  double zoom = 1.0;
  int invert = 0;

  @Override
  public void transform(FlameTransformationContext pContext, XForm pXForm, XYZPoint pAffineTP, XYZPoint pVarTP, double pAmount) {
    double x = pAffineTP.x;
    double y = pAffineTP.y;
    vec2 u = new vec2(x * zoom, y * zoom);
    vec2 k = G.Kscope(u, 0.7);
    double a = G.atan2(k.y, k.x);
    double b = G.atan(k.y, k.x);
    double c = Math.atan2(y, x);
    double color = G.fract(a / 7.0 + b * 0.1 + c * 0.01);
    pVarTP.doHide = false;
    if (invert == 0) {
      if (color > 0.5) {
        x = 0;
        y = 0;
        pVarTP.doHide = true;
      }
    } else {
      if (color <= 0.5) {
        x = 0;
        y = 0;
        pVarTP.doHide = true;
      }
    }
    pVarTP.x = pAmount * x;
    pVarTP.y = pAmount * y;
  }

  public String getName() {
    return "reg_glsl";
  }

  public String[] getParameterNames() {
    return (additionalParamNames);
  }

  public Object[] getParameterValues() {
    return (new Object[] {zoom, invert});
  }

  public void setParameter(String pName, double pValue) {
    if (pName.equalsIgnoreCase(PARAM_ZOOM)) {
      zoom = pValue;
    } else if (pName.equalsIgnoreCase(PARAM_INVERT)) {
      invert = (int) Tools.limitValue(pValue, 0, 1);
    } else
      throw new IllegalArgumentException(pName);
  }
}
