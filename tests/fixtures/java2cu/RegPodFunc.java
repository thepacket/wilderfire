package org.jwildfire.create.tina.variation;

import org.jwildfire.create.tina.base.Layer;
import org.jwildfire.create.tina.base.XForm;
import org.jwildfire.create.tina.base.XYZPoint;

import static org.jwildfire.base.mathlib.MathLib.*;

/** Synthetic regression fixture: a plain-data inner class (→ struct), a constant table
 *  (→ module-scope array), a helper mutating an object parameter (→ pointer), a chained
 *  assignment after `else`, an int parameter, per-instance state with a random initialiser. */
public class RegPodFunc extends VariationFunc {
  private static final long serialVersionUID = 1L;
  private static final String PARAM_SIDES = "sides";
  private static final String PARAM_SPREAD = "spread";
  private static final String[] paramNames = {PARAM_SIDES, PARAM_SPREAD};
  private static final double[] OFFSETS = {0.0, 0.25, 0.5, 0.75};

  private int sides = 4;
  private double spread = 0.5;
  private double phase = Math.random() * 0.1;

  private static class Pt {
    public double x, y;

    public Pt(double x, double y) {
      this.x = x;
      this.y = y;
    }
  }

  private void rotate(Pt p, double a) {
    double c = cos(a), s = sin(a);
    double x = p.x * c - p.y * s;
    p.y = p.x * s + p.y * c;
    p.x = x;
  }

  @Override
  public void transform(FlameTransformationContext pContext, XForm pXForm, XYZPoint pAffineTP, XYZPoint pVarTP, double pAmount) {
    Pt p = new Pt(pAffineTP.x, pAffineTP.y);
    int k = pContext.random(sides);
    rotate(p, 2.0 * M_PI * k / sides + phase);
    double d;
    if (k % 2 == 0)
      d = spread * OFFSETS[k & 3];
    else
      d = p.x = p.x + spread;
    pVarTP.x += pAmount * (p.x + d);
    pVarTP.y += pAmount * p.y;
  }

  @Override
  public String[] getParameterNames() {
    return paramNames;
  }

  @Override
  public Object[] getParameterValues() {
    return new Object[]{sides, spread};
  }

  @Override
  public void setParameter(String pName, double pValue) {
    if (PARAM_SIDES.equalsIgnoreCase(pName))
      sides = limitIntVal(Tools.FTOI(pValue), 1, 12);
    else if (PARAM_SPREAD.equalsIgnoreCase(pName))
      spread = pValue;
    else
      throw new IllegalArgumentException(pName);
  }

  @Override
  public String getName() {
    return "reg_pod";
  }

  @Override
  public VariationFuncType[] getVariationTypes() {
    return new VariationFuncType[]{VariationFuncType.VARTYPE_2D};
  }
}
