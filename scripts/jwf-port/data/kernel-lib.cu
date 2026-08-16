// Extracted from JWildfire's Flam4_3dKernal_TemplateJWF.cu (LGPL-2.1, (c) Andreas Maschke and contributors).
// Shared CUDA helper library that JWildfire GPU snippets call into. Transpiled on demand by cwgsl.ts.
__device__ float sqrtf_safe(float x) {
  if (x <= 0.0f)
    return 0.0f;
  else	
    return sqrtf(x);
}

__device__ float lerpf(float a, float b, float p) {
    return a + (b - a) * p;
}

__device__ float blerpf(float c00, float c10, float c01, float c11, float tx, float ty) {
    return lerpf(lerpf(c00, c10, tx), lerpf(c01, c11, tx), ty);
}

__device__ float fracf(float x) {
  return x - truncf(x);
}

__device__ int fastFloor(float f) {
  return (f >= 0 ? (int) f : (int) f - 1);
}
__device__ __constant__ float GRAD_3D_x[16] =  { 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, 0, -1, 0 };
__device__ __constant__ float GRAD_3D_y[16] =  { 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1 };
__device__ __constant__ float GRAD_3D_z[16] =  { 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 0, -1 };

// Hashing
__device__ __constant__ int X_PRIME = 1619;
__device__ __constant__ int Y_PRIME = 31337;
__device__ __constant__ int Z_PRIME = 6971;
__device__ __constant__ int W_PRIME = 1013;
__device__ int hash2D(int seed, int x, int y) {
    int hash = seed;
    hash ^= X_PRIME * x;
    hash ^= Y_PRIME * y;

    hash = hash * hash * hash * 60493;
    hash = (hash >> 13) ^ hash;

    return hash;
}

__device__ int hash3D(int seed, int x, int y, int z) {
    int hash = seed;
    hash ^= X_PRIME * x;
    hash ^= Y_PRIME * y;
    hash ^= Z_PRIME * z;

    hash = hash * hash * hash * 60493;
    hash = (hash >> 13) ^ hash;

    return hash;
}

__device__ float gradCoord3D(int seed, int x, int y, int z, float xd, float yd, float zd) {
    int hash = seed;
    hash ^= X_PRIME * x;
    hash ^= Y_PRIME * y;
    hash ^= Z_PRIME * z;

    hash = hash * hash * hash * 60493;
    hash = (hash >> 13) ^ hash;

    int idx = hash & 15;

    return xd * GRAD_3D_x[idx] + yd * GRAD_3D_y[idx] + zd * GRAD_3D_z[idx];
}
__device__ __constant__ float F3 = (float) (1.0 / 3.0);
__device__ __constant__ float G3 = (float) (1.0 / 6.0);
__device__ __constant__ float G33 =(float) ((1.0 / 6.0) * 3 - 1);
__device__ float singleSimplex(int seed, float x, float y, float z) {
    float t = (x + y + z) * F3;
    int i = fastFloor(x + t);
    int j = fastFloor(y + t);
    int k = fastFloor(z + t);

    t = (i + j + k) * G3;
    float x0 = x - (i - t);
    float y0 = y - (j - t);
    float z0 = z - (k - t);

    int i1, j1, k1;
    int i2, j2, k2;

    if (x0 >= y0) {
        if (y0 >= z0) {
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 1;
            k2 = 0;
        } else if (x0 >= z0) {
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 0;
            k2 = 1;
        } else // x0 < z0
        {
            i1 = 0;
            j1 = 0;
            k1 = 1;
            i2 = 1;
            j2 = 0;
            k2 = 1;
        }
    } else // x0 < y0
    {
        if (y0 < z0) {
            i1 = 0;
            j1 = 0;
            k1 = 1;
            i2 = 0;
            j2 = 1;
            k2 = 1;
        } else if (x0 < z0) {
            i1 = 0;
            j1 = 1;
            k1 = 0;
            i2 = 0;
            j2 = 1;
            k2 = 1;
        } else // x0 >= z0
        {
            i1 = 0;
            j1 = 1;
            k1 = 0;
            i2 = 1;
            j2 = 1;
            k2 = 0;
        }
    }

    float x1 = x0 - i1 + G3;
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + F3;
    float y2 = y0 - j2 + F3;
    float z2 = z0 - k2 + F3;
    float x3 = x0 + G33;
    float y3 = y0 + G33;
    float z3 = z0 + G33;

    float n0, n1, n2, n3;

    t = (float) 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if (t < 0) n0 = 0;
    else {
        t *= t;
        n0 = t * t * gradCoord3D(seed, i, j, k, x0, y0, z0);
    }

    t = (float) 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if (t < 0) n1 = 0;
    else {
        t *= t;
        n1 = t * t * gradCoord3D(seed, i + i1, j + j1, k + k1, x1, y1, z1);
    }

    t = (float) 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if (t < 0) n2 = 0;
    else {
        t *= t;
        n2 = t * t * gradCoord3D(seed, i + i2, j + j2, k + k2, x2, y2, z2);
    }

    t = (float) 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if (t < 0) n3 = 0;
    else {
        t *= t;
        n3 = t * t * gradCoord3D(seed, i + 1, j + 1, k + 1, x3, y3, z3);
    }

    return 32 * (n0 + n1 + n2 + n3);
}

struct __align__(8) Mat2 {
	 float a00;
	 float a01;
	 float a10;
	 float a11;
};

__device__ void Mat2_Init(Mat2 *m, float v00, float v10, float v01, float v11) {
  m->a00 = v00;
  m->a01 = v01;
  m->a10 = v10;
  m->a11 = v11;
}

__device__ void Mat2_Init(Mat2 *m, float4 v) {
  m->a00 = v.x;
  m->a10 = v.y;
  m->a01 = v.z;
  m->a11 = v.w;
}

__device__ void Mat2_Init(Mat2 *m, float2 v1, float2 v2) {
  m->a00 = v1.x;
  m->a10 = v1.y;
  m->a01 = v2.x;
  m->a11 = v2.y;
}

__device__ float2 times(Mat2 *m, float2 v){
   	return  make_float2(m->a00*v.x + m->a01*v.y , m->a10*v.x + m->a11*v.y); 
}

__device__ void add(Mat2 *m, float v) {
  m->a00 += v;
  m->a10 += v;
  m->a01 += v;
  m->a11 += v;
}

__device__ void minus(Mat2 *m, float v) {
  m->a00 -= v;
  m->a10 -= v;
  m->a01 -= v;
  m->a11 -= v;
}

__device__ void times(Mat2 *m, float v) {
  m->a00 *= v;
  m->a10 *= v;
  m->a01 *= v;
  m->a11 *= v;
}

__device__ void division(Mat2 *m, float v) {
  m->a00 /= v;
  m->a10 /= v;
  m->a01 /= v;
  m->a11 /= v;
}

struct __align__(8) Mat3 {
	 float a00;
	 float a10;
	 float a20;
	 float a01;
	 float a11;
	 float a21;
	 float a02;
	 float a12;
	 float a22;
};

__device__ void Mat3_Init(Mat3 *m, float v00, float v10, float v20, float v01, float v11, float v21, float v02, float v12, float v22 ) {
  m->a00 = v00;
  m->a10 = v10;
  m->a20 = v20;
  m->a01 = v01;
  m->a11 = v11;
  m->a21 = v21;
  m->a02 = v02;
  m->a12 = v12;
  m->a22 = v22;
}

__device__ void Mat3_Init(Mat3 *m, float3 v1, float3 v2, float3 v3) {
  m->a00 = v1.x;
  m->a10 = v1.y;
  m->a20 = v1.z;
  m->a01 = v2.x;
  m->a11 = v2.y;
  m->a21 = v2.z;
  m->a02 = v3.x;
  m->a12 = v3.y;
  m->a22 = v3.z;
}

__device__ float3 times(Mat3 *m, float3 v){
     return make_float3(m->a00*v.x + m->a01*v.y + m->a02*v.z , m->a10*v.x + m->a11*v.y + m->a12*v.z , m->a20*v.x + m->a21*v.y + m->a22*v.z);
   	
}

__device__ void add(Mat3 *m, float v) {
  m->a00 += v;
  m->a10 += v;
  m->a20 += v;
  m->a01 += v;
  m->a11 += v;
  m->a21 += v;
  m->a02 += v;
  m->a12 += v;
  m->a22 += v;
}

__device__ void minus(Mat3 *m, float v) {
  m->a00 -= v;
  m->a10 -= v;
  m->a20 -= v;
  m->a01 -= v;
  m->a11 -= v;
  m->a21 -= v;
  m->a02 -= v;
  m->a12 -= v;
  m->a22 -= v;
}

__device__ void times(Mat3 *m, float v) {
  m->a00 *= v;
  m->a10 *= v;
  m->a20 *= v;
  m->a01 *= v;
  m->a11 *= v;
  m->a21 *= v;
  m->a02 *= v;
  m->a12 *= v;
  m->a22 *= v;
}

__device__ void division(Mat3 *m, float v) {
  m->a00 /= v;
  m->a10 /= v;
  m->a20 /= v;
  m->a01 /= v;
  m->a11 /= v;
  m->a21 /= v;
  m->a02 /= v;
  m->a12 /= v;
  m->a22 /= v;
}

struct __align__(8) Mathc {
	 float a;
	 float b;
	 float c;
	 float d;
	 float e;
	 float f;	 
};

__device__ Mat3 rotEuler (float3 s) {
		float 	sa = sinf(s.x),
		ca = cosf(s.x),
		sb = sinf(s.y),
		cb = cosf(s.y),
		sc = sinf(s.z),
		cc = cosf(s.z);
		Mat3 M;
		Mat3_Init (&M,make_float3(cb*cc, -cb*sc, sb),
		              make_float3(sa*sb*cc+ca*sc, -sa*sb*sc+ca*cc, -sa*cb),
		              make_float3(-ca*sb*cc+sa*sc, ca*sb*sc+sa*cc, ca*cb)  );
	return M;
}

__device__ float distance_color(float p_red,float p_green,float p_blue,float red,float green,float blue)
{
	float dist_r = fabsf(p_red - red);
	float dist_g = fabsf(p_green - green);
	float dist_b = fabsf(p_blue - blue);
	float dist_3d_sqd = (dist_r * dist_r) + (dist_g * dist_g) + (dist_b * dist_b);
	return dist_3d_sqd;
}

__device__ float2  transfhcf (float2 xy,float a,float b,float c,float d,float e,float f)
{
  float xt=a*xy.x+b*xy.y+c;
  float yt=d*xy.x+e*xy.y+f;
  return make_float2(xt,yt);
}

__device__ float greyscale(int r,int  g,int b)
{
  int lum,red,green,blue;
  red = (r * 0.299);         
  green = (g * 0.587);         
  blue = (b * 0.114);    
  lum = red + green + blue;    
  return (float)lum/255.0f;
}

__device__ int3 dbl2int(float3 theColor)
  	{
  		int red   =  max(0, min(255, (int)floorf(theColor.x * 256.0f)));
  		int green =  max(0, min(255, (int)floorf(theColor.y * 256.0f)));
  		int blue  =  max(0, min(255, (int)floorf(theColor.z * 256.0f)));
  		return make_int3(red,green,blue);
  	}
	
	
__device__ float3  hsv2rgb (float3 c) 
	{
	  float4 K = make_float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
	  float3 p = abs(fract(make_float3(c.x,c.x,c.x)+(make_float3(K.x,K.y,K.z)))*(6.0f)-(make_float3(K.w,K.w,K.w)));
	  return mix(make_float3(K.x,K.x,K.x), clamp(p - make_float3(K.x,K.x,K.x), 0.0f, 1.0f), c.y)*c.z;
	}
	
	struct __align__(8) Jacobi_elliptic_result
{ float cn;
  float dn;
  float sn;
};
	
__device__ void Jacobi_elliptic( float uu, float emmc, Jacobi_elliptic_result *res)
{
    res->cn=0.0;
	res->dn=0.0;
	res->sn=0.0;
    
    float CA = 0.0003; 
    float a, b, c = 0.0, d = 0.0, em[13] , en[13];
    int bo;
    int l = 0;
    int ii;
    int i;
    
    
    float emc = emmc;
    float u = uu;
    if (emc != 0.0) {
      bo = 0;
      if (emc < 0.0)
        bo = 1;
      if (bo != 0) {
        d = 1.0 - emc;
        emc = -emc / d;
        d = sqrtf(d);
        u = d * u;
      }
      a = 1.0;
      res->dn = 1.0;
      
      for (i = 0; i < 8; i++) {
        l = i;
        em[i] = a;
        emc = sqrtf(emc);
        en[i] = emc;
        c = 0.5 * (a + emc);
        if (fabsf(a - emc) <= CA * a)
          break;
        emc = a * emc;
        a = c;
      }
      u = c * u;
      res->sn = sinf(u);
      res->cn = cosf(u);
      if (res->sn != 0.0) {
        a = res->cn / res->sn;
        c = a * c;
        for (ii = l; ii >= 0; --ii) {
          b = em[ii];
          a = c * a;
          c = res->dn * c;
          res->dn = (en[ii] + a) / (b + a);
          a = c / b;
        }
        a = 1.0 / sqrtf(c * c + 1.0);
        if (res->sn < 0.0)
          (res->sn) = -a;
        else
          res->sn = a;
        res->cn = c * (res->sn);
      }
      if (bo != 0) {
        a = res->dn;
        res->dn = res->cn;
        res->cn = a;
        res->sn = (res->sn) / d;
      }
    } else {
      res->cn = 1.0 / coshf(u);
      res->dn = res->cn;
      (res->sn) = tanhf(u);
    }
}

//------------- END of JS CODE--------------------------


struct __align__(8) Complex
{
  float per_fix;
  float re;
  float im;
  float save_re;
  float save_im;
};

__device__ void Complex_Init(Complex *c, float Rp, float Ip) {
  c->re = Rp;
  c->im = Ip;
  c->save_re = 0.f;
  c->save_im = 0.f;
  c->per_fix = 0.f;  
}


	
__device__ float Complex_Mag2(Complex *c) {
    return c->re * c->re + c->im * c->im;
}
  
__device__ float Complex_MagInv(Complex *c) {
    float M2 = Complex_Mag2(c);
    return (M2 < 1e-10 ? 1.0f : 1.0f / M2);
}
  
__device__ void Complex_Recip(Complex *c) {
    float mi = Complex_MagInv(c);
    c->re = c->re * mi;
    c->im = -c->im * mi;
}

__device__ void Complex_Dec(Complex *c) {
  c->re -= 1.0f;
}

__device__ void Complex_Inc(Complex *c) {
  c->re += 1.0f;
}

__device__ void Complex_Neg(Complex *c) {
  c->re = -c->re;
  c->im = -c->im;
}
  
__device__ void Complex_Div(Complex *c, Complex *zz) {
  float r2 = c->im * zz->im + c->re * zz->re;
  float i2 = c->im * zz->re - c->re * zz->im;
  float M2 = Complex_MagInv(zz);
  c->re = r2 * M2;
  c->im = i2 * M2;
}
  
  __device__ void Complex_DivR(Complex *T,Complex *zz) {
	float r2 = zz->im * T->im + zz->re * T->re;
	float i2 = zz->im * T->re - zz->re * T->im;
	float M2 = Complex_MagInv(T);
	T->re = r2 * M2;
	T->im = i2 * M2;
} 

__device__ void Complex_Copy(Complex *c, Complex *zz) {
  c->re = zz->re;
  c->im = zz->im;
}
  
__device__ float Complex_Mag2eps(Complex *c) {
    return c->re * c->re + c->im * c->im + 1e-10;
}

__device__ float Complex_Arg(Complex *c) {
  return (c->per_fix + atan2f(c->im, c->re));
}

__device__ void Complex_Log(Complex *c) {
  Complex L_eps;
  Complex_Init(&L_eps, 0.5f * logf(Complex_Mag2eps(c)), Complex_Arg(c));
  Complex_Copy(c, &L_eps);
}

__device__ void Complex_Scale(Complex *c, float mul) {
    c->re = c->re * mul;
    c->im = c->im * mul;
}
  
__device__ void Complex_AtanH(Complex *c) {
    Complex D;
	Complex_Init(&D, c->re, c->im);
    Complex_Dec(&D);
    Complex_Neg(&D);
    Complex_Inc(c);
    Complex_Div(c, &D);
    Complex_Log(c);
    Complex_Scale(c, 0.5f);
}

__device__ void Complex_AcotH(Complex *c) {
   Complex_Recip(c);
   Complex_AtanH(c);
}

__device__ void Complex_Flip(Complex *c) {
    float r2 = c->im;
    float i2 = c->re;
    c->re = r2;
    c->im = i2;
  }
  
__device__ void Complex_Sqr(Complex *c) {
  float r2 = c->re * c->re - c->im * c->im;
  float i2 = 2.f * c->re * c->im;
  c->re = r2;
  c->im = i2;
}  

  
__device__ void Complex_Add(Complex *c, Complex *zz) {
  c->re += zz->re;
  c->im += zz->im;
}

__device__ void Complex_Sub(Complex *c, Complex *zz) {
  c->re -= zz->re;
  c->im -= zz->im;
}


__device__ void Complex_Mul(Complex *c, Complex *zz) {
   if (zz->im == 0.0) {
      Complex_Scale(c, zz->re);
      return;
   }
   float  r2 = c->re * zz->re - c->im * zz->im;
   float  i2 = c->re * zz->im + c->im * zz->re;
   c->re = r2;
   c->im = i2;
}
    
  
__device__ void Complex_One(Complex *c) {
  c->re = 1.0f;
  c->im = 0.0f;
}

__device__ void Complex_Conj(Complex *c) {
  c->im = -c->im;
}


__device__ float Complex_Radius(Complex *c) {
    return hypotf(c->re, c->im);
}

__device__ void Complex_Sqrt(Complex *c) {
  float Rad = Complex_Radius(c);
  float sb = (c->im < 0) ? -1.f : 1.f;
  c->im = sb * sqrtf(0.5f * (Rad - c->re));
  c->re = sqrtf(0.5f * (Rad + c->re));
  if (c->per_fix < 0)
    Complex_Neg(c);
}
  
  
__device__ void Complex_ToP(Complex *c, Complex *dst) {
  Complex_Init(dst, Complex_Radius(c), Complex_Arg(c));
}
  
  
__device__ void Complex_UnP(Complex *c, Complex *dst) {
  Complex_Init(dst, c->re * cosf(c->im), c->re * sinf(c->im));
}  
  
__device__ void Complex_Pow(Complex *c, float exp) {
    if (exp == 0.0f) {
      Complex_One(c);
      return;
    }
    float ex = fabsf(exp);
    if (exp < 0) {
      Complex_Recip(c);
    }
    if (ex == 0.5f) {
      Complex_Sqrt(c);
      return;
    }
    if (ex == 1.0f) {
      return;
    }
    if (ex == 2.0f) {
      Complex_Sqr(c);
      return;
    }
    // In general we need sin, cos etc
    Complex PF;
    Complex_ToP(c, &PF);
    PF.re = powf(PF.re, ex);
    PF.im = PF.im * ex;
	
	Complex PFU;	
	Complex_UnP(&PF, &PFU);	
    Complex_Copy(c, &PFU);
  }
  
 
__device__ void Complex_AsinH(Complex *c) {
  Complex D;
  Complex_Init(&D, c->re, c->im);
  Complex_Sqr(&D);
  Complex_Inc(&D);
  Complex_Pow(&D, 0.5f);
  Complex_Add(c, &D);
  Complex_Log(c);
}

__device__ void Complex_AsecH(Complex *c) {
   Complex_Recip(c);
   Complex_AsinH(c);
}

__device__ void Complex_Exp(Complex *c) {
   c->re = expf(c->re);
   Complex unp;
   Complex_UnP(c, &unp);
   Complex_Copy(c, &unp);
}

__device__ void Complex_AcosH(Complex *c) {
  Complex D;
  Complex_Init(&D, c->re, c->im);
  Complex_Sqr(&D);
  Complex_Dec(&D);
  Complex_Pow(&D, 0.5f);
  Complex_Add(c, &D);
  Complex_Log(c);
}

__device__ void Complex_AcosecH(Complex *c) {
   Complex_Recip(c);
   Complex_AcosH(c);
}

__device__ void Complex_SinH(Complex *c) {
    float rr = 0.0;
    float ri = 0.0;
    float er = 1.0;
    c->re = expf(c->re);
    er /= c->re;
    rr = 0.5 * (c->re - er);
    ri = rr + er;
    c->re = cosf(c->im) * rr;
    c->im = sinf(c->im) * ri;
}
  
__device__ void Complex_CosH(Complex *c) {
    float rr = 0.0;
    float ri = 0.0;
    float er = 1.0;
    c->re = expf(c->re);
    er /= c->re;
    rr = 0.5 * (c->re - er);
    ri = rr + er;
    c->re = cosf(c->im) * ri;
    c->im = sinf(c->im) * rr;
}

__device__ void Complex_Sin(Complex *c) {
    Complex_Flip(c);
    Complex_SinH(c);
    Complex_Flip(c);
}

__device__ void Complex_Cos(Complex *c) {
    Complex_Flip(c);
    Complex_CosH(c);
    Complex_Flip(c);
}

__device__ void Complex_Asin(Complex *c) {
    Complex_Flip(c);
    Complex_AsinH(c);
    Complex_Flip(c);
}

__device__ void Complex_Acos(Complex *c) {
    Complex_Flip(c);
    Complex_AsinH(c);
    Complex_Flip(c); 
    c->re = (M_PI_F/2.0) - (c->re);
    c->im = -(c->im); 
}

__device__ void Complex_Atan(Complex *c) { 
    Complex_Flip(c);
    Complex_AtanH(c);
    Complex_Flip(c);
} 


// Additional complex Functions

__device__ float Complex_arg (Complex z) {
    float result;
    result = atan2f(z.im, z.re);
    return result;
  }
  
__device__ float Complex_norm(Complex z) {
    double u = z.re;
    double v = z.im;
    return (u * u + v * v);
  }
  
__device__ float Complex_mag (Complex z) {
    return sqrtf(z.re*z.re + z.im*z.im);
 }
__device__ Complex Complex_plus (Complex a,Complex z) {
   Complex tmp;
   Complex_Init(&tmp, a.re+ z.re, a.im + z.im);
   return tmp;
  }
  
__device__ Complex Complex_minus (Complex a,Complex z) {
   Complex tmp;
   Complex_Init(&tmp, a.re - z.re, a.im - z.im);
   return tmp;
  }

__device__ Complex Complex_times (Complex a, float x) {
    Complex tmp;
    Complex_Init(&tmp,x*a.re,x*a.im);
	return tmp;
}

__device__ Complex Complex_times (Complex a, Complex z) {
   Complex tmp;
   Complex_Init(&tmp, a.re*z.re - a.im*z.im,a.re*z.im + a.im*z.re);
   return tmp;
}

__device__ Complex Complex_divideBy (Complex a, Complex z) {
    Complex tmp;
    float rz = Complex_mag(z);
    if(fabsf(rz) > 1.0e-12)
    {
	  Complex_Init(&tmp,(a.re * z.re + a.im * z.im)/(rz * rz),
                        (a.im * z.re - a.re * z.im)/(rz * rz));
    }	
	return tmp;
}

__device__ Complex Complex_sqrt(Complex z) {
    Complex tmp;
	float r = sqrtf(Complex_mag(z));
    float phi = Complex_arg(z)/2.0;
	Complex_Init(&tmp,r*cosf(phi),r*sinf(phi));
	return tmp;
}

__device__ Complex Complex_ln(Complex z) {
    Complex tmp;
    float rr = logf(Complex_mag(z))/logf(2.718);
    float ii = Complex_arg(z);
    Complex_Init(&tmp,rr,ii);
	return tmp;
}
  
__device__ Complex Complex_sin(Complex z) { 
    float r = sinf(z.re) * coshf(z.im);
    float i = cosf(z.re) * sinhf(z.im);
	Complex tmp;
	Complex_Init(&tmp,r,i);
    return tmp;
  }

__device__ Complex Complex_asinh(Complex zz)  {
    Complex i,z;
	Complex_Init(&i,1.0,0.0);
	z = Complex_plus(i,Complex_times(zz,zz));
    z = Complex_sqrt(z);
    z = Complex_plus(zz,z);
    z = Complex_ln(z);
    return z;
}

__device__ Complex Complex_asin(Complex z) {
    Complex j,zz;
	Complex_Init(&j,0.0, 1.0);
	Complex one;
	Complex_Init(&one,1.0,0.0);
	zz = Complex_minus(one , Complex_times(z,z));
    zz = Complex_sqrt(zz);
    zz = Complex_plus(zz,Complex_times(j,z));
    zz = Complex_times(Complex_times(j,Complex_ln(zz)), -1.0);
    return zz;
  }

__device__ Complex Complex_acos(Complex z) {
      Complex i,j,zz;
	  Complex_Init(&i,1.0,0.0);
	  Complex_Init(&j,0.0,1.0);
	  zz=Complex_minus(Complex_times(z,z),i);
      zz = Complex_sqrt(zz);
      zz = Complex_plus(z,zz);
      zz = Complex_times(Complex_times(j,Complex_ln(zz)),-1.0);
      return zz;
}

__device__ Complex Complex_tan(Complex z) {
    Complex tmp;
    float nenner = cosf(2.*z.re) + coshf(2*z.im);
    float r = sinf(2.*z.re) / nenner;
    float i = sinhf(2.*z.im) / nenner;
	Complex_Init(&tmp,r,i);
    return tmp;;
}


  

__device__ float sqrf(float x) {
  return x*x;
} 
