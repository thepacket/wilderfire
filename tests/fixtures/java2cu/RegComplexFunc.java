package org.jwildfire.create.tina.variation;

import org.jwildfire.base.mathlib.Complex;
import org.jwildfire.create.tina.base.XForm;
import org.jwildfire.create.tina.base.XYZPoint;

import static org.jwildfire.base.mathlib.MathLib.*;

/** Synthetic regression fixture: JWildfire's Complex class (mutating methods, copy constructor), a helper that
 *  only hands its object parameter on to another helper's pointer parameter, and a null-guarded block. */
public class RegComplexFunc extends VariationFunc {
  private static final long serialVersionUID = 1L;
  private static final String PARAM_RE = "re";
  private static final String PARAM_IM = "im";
  private static final String[] paramNames = {PARAM_RE, PARAM_IM};

  private double re = 1.0;
  private double im = 0.5;

  private static class Pair {
    public double a, q;
  }

  private void fill(Pair out, double v) {
    out.a = v;
    out.q = -v;
  }

  private void relay(Pair q, double v) {
    fill(q, v * 2.0);
  }

  @Override
  public void transform(FlameTransformationContext pContext, XForm pXForm, XYZPoint pAffineTP, XYZPoint pVarTP, double pAmount) {
    Complex z = new Complex(pAffineTP.x, pAffineTP.y);
    Complex c = new Complex(re, im);
    Complex w = new Complex(z);
    z.Exp();
    z.Div(c);
    if (im != 0) {
      z.Sqr();
      z.Mul(c);
    }
    z.Add(w);
    z.Log();
    z.Scale(pAmount);
    Pair p = new Pair();
    relay(p, z.re);
    if (p == null || c == null) {
      p.a = 0;
    }
    pVarTP.x += z.re + p.a * 0.0;
    pVarTP.y += z.im + p.q * 0.0;
  }

  @Override
  public String[] getParameterNames() {
    return paramNames;
  }

  @Override
  public Object[] getParameterValues() {
    return new Object[]{re, im};
  }

  @Override
  public void setParameter(String pName, double pValue) {
    if (PARAM_RE.equalsIgnoreCase(pName)) {
      double v = Math.max(1e-9, pValue);
      re = v;
    } else if (PARAM_IM.equalsIgnoreCase(pName))
      im = pValue;
    else
      throw new IllegalArgumentException(pName);
  }

  @Override
  public String getName() {
    return "reg_complex";
  }
}
