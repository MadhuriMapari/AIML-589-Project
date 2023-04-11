STAMP=`ls -ctr run_* | tail -n 1 | sed 's/run_//g'`
echo STAMP=$STAMP
for pb in `cat listpb.txt`
do
for k in 1 2
do
STAMP2=`ls -ctr results/*-${pb}-*/*.csv | tail -n $k | head -n 1 | sed 's/.*STAMP/STAMP/g' | sed 's/-[\.0-9]*.log.csv//g'`
#ls results/*-${pb}-*${STAMP}*/*.csv
echo $pb ":" `tail -n 1 results/*-${pb}-*${STAMP2}*/*.csv | awk -F ',' '{print $4}' | grep '[0-9]'` vs expected `grep $pb table3.txt | awk '{print $2}'`
done
done
