#include <math.h>
#include <complex.h>

//#include "cvis.h"

double Vud(double X)
  /* X = pi*c*B*diam/wavel
  B in m, diam in mas, wavel in um
  c = 0.004848135802469136
  */
{
    double res;
    res =  1.0;
    if (X>0) {
      res += -pow(X,2)/8.0 + pow(X,4)/192.0;
    }
    if (X>2) {
      res += - pow(X,6)/9216.0
            + pow(X,8)/737280.0
            - pow(X,10)/88473600.0
            + pow(X,12)/14863564800.0
            - pow(X,14)/3329438515200.0;
    }
    return res;
}

double complex VbinMonoSingle(double u, double v, double wl, double x, double y, double fc, double fres, double diam, double diamc)
/*
  u, v: in m
  wl: in um
  x, y, diam, diamc: in mas
  fc, fres: in % of the primary
*/
{
    double B, Vs, Vc, phi;
    B = sqrt(u*u+v*v);
    phi = -0.0304617*(u*x + v*y)/wl;
    Vs = Vud(0.015231*B*diam/wl);
    Vc = Vud(0.015231*B*diamc/wl);
    return (Vs + fc/100.*Vc*cexp(phi*I))/(1 + fc/100 + fres/100);
}

void VbinMono(int n, double complex *V, double *u, double *v, double *wl,
               double x, double y, double fc, double fres, double diam, double diamc)
/*
  u, v: in m
  wl: in um
  x, y, diam, diamc: in mas
  fc, fres: in % of the primary
*/
{
    for (int i=0; i<n;i++){
      V[i] = VbinMonoSingle(u[i],  v[i],  wl[i],  x,  y,  fc,  fres,  diam,  diamc);
    }
}

void Vbin(int n, double complex *V, double *u, double *v, double *wl, double *dwl, int nsmear,
          double x, double y, double fc, double fres, double diam, double diamc)
/*
  u, v: in m
  wl: in um
  x, y, diam, diamc: in mas
  fc, fres: in % of the primary
*/
{
    double B, Vs, Vc, phi;
    for (int i=0; i<n;i++){
      B = sqrt(u[i]*u[i]+v[i]*v[i]);
      Vs = Vud(0.015231*B*diam/wl[i]);
      Vc = Vud(0.015231*B*diamc/wl[i]);
      phi = -0.0304617*(u[i]*x + v[i]*y);
      V[i] = Vs;
      if (nsmear>2){
        for (double j=0; j<nsmear; j++ ){
          V[i] +=  fc/100.*Vc*cexp(I*phi/(wl[i] + (j/(nsmear-1)-0.5)*dwl[i]))/nsmear;
        }
      } else {
        V[i] += fc/100.*Vc*cexp(I*phi/wl[i]);
      }
      V[i] /= (1.0 + fc/100. + fres/100.);
  }
}
