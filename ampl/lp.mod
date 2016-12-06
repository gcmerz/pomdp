set S = {1..3};

param numAlpha;
set alphaI = {1..numAlpha};

param diff {i in alphaI, s in S};

var sigma;
var pi {i in S} >= 0;

maximize sigmaValue: sigma;

subject to distribution:
    sum {i in S} pi[i] = 1;

subject to optimal {i in alphaI}:
    -1*sigma + sum {s in S} (pi[s]*diff[i,s]) >= 0;
