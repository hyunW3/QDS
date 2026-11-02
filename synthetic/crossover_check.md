relative MSE (dB = -10log10); 'rot' = real Hadamard applied to Laplace blocks
   G |                       E2M1 |         uniform(E1M2/INT4)
   4 | native 24.32 rot 22.60 (native ) | native 24.37 rot 25.48 (rot wins)
   8 | native 22.22 rot 21.27 (native ) | native 21.84 rot 23.30 (rot wins)
  16 | native 20.80 rot 20.43 (native ) | native 19.89 rot 21.72 (rot wins)
  32 | native 19.79 rot 19.88 (rot wins) | native 18.34 rot 20.47 (rot wins)
  64 | native 19.00 rot 19.53 (rot wins) | native 17.03 rot 19.52 (rot wins)
 128 | native 18.35 rot 19.28 (rot wins) | native 15.89 rot 18.73 (rot wins)
 256 | native 17.77 rot 19.08 (rot wins) | native 14.90 rot 18.01 (rot wins)
E2M1: rotation first wins at G = 32
uniform(E1M2/INT4): rotation first wins at G = 4

G=16 error share by normalized region (t = |x|/block absmax)
  E2M1                 native   dB 20.79 | dead-zone t<qmin/2  6.8%, top two bins 47.7%, middle 45.5%
  E2M1                 rotated  dB 20.43 | dead-zone t<qmin/2  1.8%, top two bins 66.6%, middle 31.6%
  uniform(E1M2/INT4)   native   dB 19.91 | dead-zone t<qmin/2 25.0%, top two bins  4.4%, middle 70.6%
  uniform(E1M2/INT4)   rotated  dB 21.71 | dead-zone t<qmin/2 12.2%, top two bins 10.4%, middle 77.5%
