alpha=(0.0 0.5 1.0)
k_line=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
cost=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
len_alpha=${#alpha[@]}
len_cost=${#cost[@]}
len_k_line=${#k_line[@]}
alpha1=$((len_alpha +1))
i=$(( ($1 / $len_k_line / $len_cost) % $alpha1 ))
j=$(( ($1 / $len_cost)% $len_k_line))
k=$(($1 % $len_cost))
if [ $i -eq 3 ]; then
    python3 cifar_posthoc_deferral.py --k-line ${k_line[$j]} --alpha 0.0 --cost ${cost[$k]}
else
    python3 cifar_posthoc_deferral.py --k-line ${k_line[$j]} --alpha ${alpha[$i]} --cost ${cost[$k]} --is-defer 
fi
