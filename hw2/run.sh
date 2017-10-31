PYTHON=python3
MAIN=policy_gradient.py
ENV=InvertedPendulum-v1
ITERS=500
ENUM=10
$PYTHON $MAIN $ENV -n $ITERS -b 1000 -e $ENUM -dna --exp_name sb_no_rtg_dna
$PYTHON $MAIN $ENV -n $ITERS -b 1000 -e $ENUM -rtg -dna --exp_name sb_rtg_dna
$PYTHON $MAIN $ENV -n $ITERS -b 1000 -e $ENUM -rtg --exp_name sb_rtg_na
$PYTHON $MAIN $ENV -n $ITERS -b 5000 -e $ENUM -dna --exp_name lb_no_rtg_dna
$PYTHON $MAIN $ENV -n $ITERS -b 5000 -e $ENUM -rtg -dna --exp_name lb_rtg_dna
$PYTHON $MAIN $ENV -n $ITERS -b 5000 -e $ENUM -rtg --exp_name lb_rtg_na
