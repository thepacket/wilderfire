package org.jwildfire.create.tina.variation;

import org.jwildfire.create.tina.base.Layer;
import org.jwildfire.create.tina.base.XForm;
import org.jwildfire.create.tina.base.XYZPoint;

import static org.jwildfire.base.mathlib.MathLib.*;

/** Synthetic regression fixture: a JWildfire "prepost" pair — invtransform() rewrites the
 *  affine point before the other variations run (priority -2), transform() maps the
 *  output back (priority 2). Also a helper taking pAffineTP (inlined) and an init() precalc. */
public class RegPrePostFunc extends VariationFunc {
  private static final long serialVersionUID = 1L;
  private static final String PARAM_SCALE = "scale";
  private static final String PARAM_ANGLE = "angle";
  private static final String[] paramNames = {PARAM_SCALE, PARAM_ANGLE};

  private double scale = 1.5;
  private double angle = 30.0;
  private double sina, cosa;

  @Override
  public void init(FlameTransformationContext pContext, Layer pLayer, XForm pXForm, double pAmount) {
    sina = sin(angle * M_PI / 180.0);
    cosa = cos(angle * M_PI / 180.0);
  }

  private void moveInput(XYZPoint pAffineTP, double s) {
    double x = pAffineTP.x * s, y = pAffineTP.y * s;
    pAffineTP.x = x * cosa - y * sina;
    pAffineTP.y = x * sina + y * cosa;
  }

  @Override
  public void invtransform(FlameTransformationContext pContext, XForm pXForm, XYZPoint pAffineTP, XYZPoint pVarTP, double pAmount) {
    moveInput(pAffineTP, 1.0 / scale);
  }

  @Override
  public void transform(FlameTransformationContext pContext, XForm pXForm, XYZPoint pAffineTP, XYZPoint pVarTP, double pAmount) {
    double x = pVarTP.x * scale, y = pVarTP.y * scale;
    pVarTP.x = x * cosa + y * sina;
    pVarTP.y = -x * sina + y * cosa;
  }

  @Override
  public String[] getParameterNames() {
    return paramNames;
  }

  @Override
  public Object[] getParameterValues() {
    return new Object[]{scale, angle};
  }

  @Override
  public void setParameter(String pName, double pValue) {
    if (PARAM_SCALE.equalsIgnoreCase(pName))
      scale = pValue;
    else if (PARAM_ANGLE.equalsIgnoreCase(pName))
      angle = pValue;
    else
      throw new IllegalArgumentException(pName);
  }

  @Override
  public String getName() {
    return "reg_prepost";
  }

  @Override
  public int getPriority() {
    return 2;
  }

  @Override
  public VariationFuncType[] getVariationTypes() {
    return new VariationFuncType[]{VariationFuncType.VARTYPE_2D, VariationFuncType.VARTYPE_PREPOST};
  }
}
