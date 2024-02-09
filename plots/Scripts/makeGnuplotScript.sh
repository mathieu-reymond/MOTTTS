#!/bin/bash

nargs=$#
# Arguments should be even and >= 4 (unless you use skips)
# cum is whether to plot the cumulative data or the per-timestep data
if [ $nargs -lt 3 ]
then
    echo "Usage: $0 output xrange xtics yrange ytics title file label cum [file label cum...]"
    echo "       Instead of [file label cum] you can also use a single 'skip' to skip the color that would be used"
    exit 0
fi

KEYLW=1

output=$1
shift

if [ "$output" = "term" ]; then
    printf "set terminal wxt size 1200,900 linewidth 7 rounded\n"
    printf "set key font \",20\"\n"
    printf "set xtics font \", 24\"\n"
    printf "set ytics font \", 24\"\n"
    printf "set lmargin 15\n"
    printf "set rmargin 6\n"
    printf "set tmargin 3\n"
    printf "set bmargin 5\n"
    printf "set xtics offset 0,-2,0\n"
else
    printf "set output '$output'\n"

    extension="${output##*.}"
    if [ "$extension" = "png" ]; then
        printf "set terminal pngcairo size 1200,900 enhanced font \"verdana\" linewidth 7 rounded\n"
        printf "set key font \",20\"\n"
        printf "set xtics font \", 24\"\n"
        printf "set xtics offset 0,-0.8,0\n"
        printf "set ytics font \", 24\"\n"
        printf "set lmargin 15\n"
        printf "set rmargin 6\n"
        printf "set tmargin 3\n"
        printf "set bmargin 5\n"
        printf "set key reverse Left left bottom samplen 2.5 box opaque width -5\n"
    elif [ "$extension" = "pdf" ]; then
        printf "set terminal pdfcairo size 12,9 enhanced font \"Liberation Sans\" linewidth 7 rounded\n"
        printf "set key font \",50\"\n"
        printf "set xtics font \", 40\"\n"
#        printf "set ytics font \", 40\"\n"
        printf "set y2tics font \", 40\"\n"
        printf "unset ytics\n"
#         printf "set lmargin 20\n"
#         printf "set rmargin 9\n"
        printf "set lmargin 3\n"
        printf "set rmargin 21\n"
        printf "set tmargin 3\n"
        printf "set bmargin 5\n"
        printf "set xtics offset 0,-2,0\n"
        printf "set key reverse Left top left samplen 2.25 box lw 0.1 opaque width 1 height +0.5\n"
    else
        echo "Extension $extension not supported"
        exit 1
    fi
fi

# Define dash types by scratch so we can have the legend with shorter ones.
# These styles try to emulate the default ones
# Note that we ignore 1, 6, 11, 16, ... since those are by default solid lines which are fine.
printf "set dashtype 2 (30,30)\n"
printf "set dashtype 12 (9,9)\n"
printf "set dashtype 3 (7,15)\n"
printf "set dashtype 13 (4,7)\n"
printf "set dashtype 4 (10,15,30,15)\n"
printf "set dashtype 14 (5,7,15,7)\n"
printf "set dashtype 5 (30,15,5,15,5,15)\n"
printf "set dashtype 15 (10,6,2,6,2,5)\n"
# 6, 16 ignored...
# printf "set dashtype 7 (30,30)\n"
# printf "set dashtype 17 (9,9)\n"
# printf "set dashtype 8 (7,15)\n"
# printf "set dashtype 18 (4,7)\n"
# printf "set dashtype 9 (10,15,30,15)\n"
# printf "set dashtype 19 (5,7,15,7)\n"
# printf "set dashtype 10 (30,15,5,15,5,15)\n"
# printf "set dashtype 20 (10,6,2,6,2,5)\n"


xrange=$1
shift
xtics=$1
shift
yrange=$1
shift
ytics=$1
shift

if [ "$xtics" = "na" ]; then
    xtics=10000
else
    printf "set xtics $xtics\n"
fi

xmax=0
if [ "$xrange" = "na" ]; then
    printf "set xrange [0<*:]\n"
elif [[ "$xrange" =~ ":" ]]; then
    printf "set xrange [$xrange]\n"
    # Split into two values
    arr=(${xrange//:/ })
    # Abs value
    arr0=${arr[0]#-}
    arr1=${arr[1]#-}
    # Pick largest
    xmax=$(( arr0 > arr1 ? arr0 : arr1 ))
else
    printf "set xrange [0:$xrange]\n"
    xmax=$xrange
fi

if [ $xtics -ge 99999 ] || [ $xmax -ge 99999 ]; then
    printf "set format x \"%%.0t*10^%%T\"\n"
    printf "set xtics add ('0^' 0)\n"
fi

if [ "$ytics" = "na" ]; then
    ytics=10000
else
    printf "set y2tics $ytics\n"
fi

ymax=0
if [ "$yrange" = "na" ]; then
    printf ""
elif [[ "$yrange" =~ ":" ]]; then
    printf "set yrange [$yrange]\n"
    printf "set y2range [$yrange]\n"
    # Split into two values
    arr=(${yrange//:/ })
    # Abs value
    arr0=${arr[0]#-}
    arr1=${arr[1]#-}
    # Pick largest
    ymax=$(( arr0 > arr1 ? arr0 : arr1 ))
else
    printf "set y2range [0:$yrange]\n"
    printf "set yrange [0:$yrange]\n"
    ymax=$yrange
fi

if [ $ytics -ge 99999 ] || [ $ymax -ge 99999 ]; then
    printf "set format y \"%%.0t*10^%%T\"\n"
    printf "set format y2 \"%%.1t*10^%%T\"\n"
    printf "set y2tics add ('0^' 0)\n"
fi

printf "set style fill transparent solid 0.4 noborder\n"
printf "set colorsequence podo\n"
printf "set datafile missing NaN\n"

title=$1
shift

# title and file setup
printf "set title \"$title\"\n"

COUNTER=0
while [ "$1" = "skip" ]
do
    COUNTER=$((COUNTER+1))
    shift
done

# Setup for required first file (filename, title)
main=2
off=4
if [[ "$3" =~ "1" ]]; then # If cumulative
    ((main++))
    ((off++))
fi

printf "plot '$1' using 1:$main with line smooth csplines ls $((COUNTER+1)) dt $((COUNTER%5+1)) notitle axes x1y2, \\"
printf "\n   NaN with line title '$2' ls $((COUNTER+1)) dt $((COUNTER%5+11)) lw $KEYLW axes x1y2"
if [[ "$3" != *"na" ]]; then
    printf ", \\"
    printf "\n   '$1' using 1 : (\$$main-\$$off) : (\$$main+\$$off) with filledcurves ls $((COUNTER+1)) notitle axes x1y2"
fi
shift
shift
shift

# Iterate over all remaining arguments
while [ $# -ne 0 ]
do
    main=2
    off=4
    if [[ "$3" =~ "1" ]]; then # If cumulative
        ((main++))
        ((off++))
    fi

    COUNTER=$((COUNTER+1))

    if [ "$1" = "skip" ]; then
        shift
        continue
    fi

    printf ", \\"
    printf "\n   '$1' using 1:$main with line smooth csplines ls $((COUNTER+1)) dt $((COUNTER%5+1)) notitle axes x1y2, \\"
    printf "\n   NaN with line title '$2' ls $((COUNTER+1)) dt $((COUNTER%5+11)) lw $KEYLW axes x1y2"
    if [[ "$3" != *"na" ]]; then
        printf ", \\"
        printf "\n   '$1' using 1 : (\$$main-\$$off) : (\$$main+\$$off) with filledcurves ls $((COUNTER+1)) notitle axes x1y2"
    fi
    shift
    shift
    shift
done

# Remember to launch gnuplot with the -p option to see the plots!
printf "\n"

# Prevent stupid plot from closing
if [ "$output" = "term" ]; then
    printf "pause -1\n"
fi
