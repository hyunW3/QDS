
layers.7.attention.wq fwd_x (H16): 327680 blocks
  option  | share  | mean #small (t<1/4) | mean #large (t>2/3) | MSE gain vs @6
  E2M1@6  |  51.7% |   6.60 / 16 |   2.49 / 16 |   0.0%
  E2M1@4  |  40.8% |   5.84 / 16 |   3.70 / 16 |  30.5%
  FLIP@6  |   7.5% |   1.92 / 16 |   7.25 / 16 |  60.5%

layers.7.feed_forward.w2 fwd_x (H16): 917504 blocks
  option  | share  | mean #small (t<1/4) | mean #large (t>2/3) | MSE gain vs @6
  E2M1@6  |  45.9% |   6.04 / 16 |   2.77 / 16 |   0.0%
  E2M1@4  |  41.4% |   5.42 / 16 |   4.04 / 16 |  31.3%
  FLIP@6  |  12.7% |   1.71 / 16 |   6.97 / 16 |  57.6%

layers.7.feed_forward.w2 dgrad_dy (H16): 327680 blocks
  option  | share  | mean #small (t<1/4) | mean #large (t>2/3) | MSE gain vs @6
  E2M1@6  |  53.8% |   6.69 / 16 |   2.46 / 16 |   0.0%
  E2M1@4  |  41.9% |   5.92 / 16 |   3.66 / 16 |  30.4%
  FLIP@6  |   4.3% |   3.08 / 16 |   5.39 / 16 |  39.4%

layers.7.feed_forward.w2 wgrad_dyT (H16): 327680 blocks
  option  | share  | mean #small (t<1/4) | mean #large (t>2/3) | MSE gain vs @6
  E2M1@6  |  51.5% |   6.37 / 16 |   2.61 / 16 |   0.0%
  E2M1@4  |  43.1% |   5.72 / 16 |   3.82 / 16 |  30.7%
  FLIP@6  |   5.4% |   2.98 / 16 |   5.46 / 16 |  40.0%

layers.7.attention.wq dgrad_dy (H16): 327680 blocks
  option  | share  | mean #small (t<1/4) | mean #large (t>2/3) | MSE gain vs @6
  E2M1@6  |  52.5% |   6.51 / 16 |   2.53 / 16 |   0.0%
  E2M1@4  |  42.4% |   5.76 / 16 |   3.75 / 16 |  30.6%
  FLIP@6  |   5.1% |   2.97 / 16 |   5.44 / 16 |  39.8%

layers.7.attention.wq wgrad_dyT (H16): 327680 blocks
  option  | share  | mean #small (t<1/4) | mean #large (t>2/3) | MSE gain vs @6
  E2M1@6  |  46.1% |   6.02 / 16 |   2.82 / 16 |   0.0%
  E2M1@4  |  43.1% |   5.43 / 16 |   4.07 / 16 |  31.4%
  FLIP@6  |  10.7% |   2.03 / 16 |   6.58 / 16 |  53.2%
