package org.jwildfire.create.tina.variation;

import org.jwildfire.create.tina.base.Layer;
import org.jwildfire.create.tina.base.XForm;
import org.jwildfire.create.tina.base.XYZPoint;

import static org.jwildfire.base.mathlib.MathLib.*;

/** Synthetic regression fixture in JWildfire's variation style (params, setParameter replay,
 *  a static-final constant, a long seed, a helper drawing randoms, preserve-z, i++ in a condition). */
public class RegDemoFunc extends VariationFunc {
  private static final long serialVersionUID = 1L;
  private static final String PARAM_POWER = "power";
  private static final String PARAM_DIST = "dist";
  private static final String PARAM_SEED = "seed";
  private static final String[] paramNames = {PARAM_POWER, PARAM_DIST, PARAM_SEED};
  private static final double SCALE = 0.75;

  private double power = 2.0;
  private double dist = 1.0;
  private long seed = 12345;
  private double invPower;

  @Override
  public void init(FlameTransformationContext pContext, Layer pLayer, XForm pXForm, double pAmount) {
    invPower = 1.0 / power;
  }

  @Override
  public void transform(FlameTransformationContext pContext, XForm pXForm, XYZPoint pAffineTP, XYZPoint pVarTP, double pAmount) {
    double a = atan2(pAffineTP.y, pAffineTP.x) * invPower;
    int n = 0;
    double r = sqrt(pAffineTP.x * pAffineTP.x + pAffineTP.y * pAffineTP.y);
    if (n++ < 1 && r > dist) {
      r = jitter(pContext, r) * SCALE;
    }
    double sn = sin(a), cs = cos(a);
    pVarTP.x += pAmount * r * cs;
    pVarTP.y += pAmount * r * sn;
    if (pContext.isPreserveZCoordinate()) {
      pVarTP.z += pAmount * pAffineTP.z;
    }
    seed = seed * 1103515245L + 1;
  }

  private double jitter(FlameTransformationContext pContext, double r) {
    return r + (pContext.random() - 0.5) * 0.01 * dist;
  }

  @Override
  public String[] getParameterNames() {
    return paramNames;
  }

  @Override
  public Object[] getParameterValues() {
    return new Object[]{power, dist, seed};
  }

  @Override
  public void setParameter(String pName, double pValue) {
    if (PARAM_POWER.equalsIgnoreCase(pName))
      power = pValue;
    else if (PARAM_DIST.equalsIgnoreCase(pName))
      dist = pValue;
    else if (PARAM_SEED.equalsIgnoreCase(pName))
      seed = (long) pValue;
    else
      throw new IllegalArgumentException(pName);
  }

  @Override
  public String getName() {
    return "reg_demo";
  }

  @Override
  public VariationFuncType[] getVariationTypes() {
    return new VariationFuncType[]{VariationFuncType.VARTYPE_2D};
  }
}
