lang=$1
dir="/output/deep_srl/${lang}_psd_new"
basen=$(basename $dir)
testfile=~/data/conll2009languages/test/${lang}/test.${lang}.conll

echo $HOME
echo ~
ls ~/data/conll2009languages/test/${lang}/
ls -l $testfile
#exit

python -m allennlp.run evaluate --archive_file ${dir}/model.tar.gz --evaluation_data_file $testfile --print_predictions ${basen}.predictions.conll --cuda_device 0
perl scripts/eval09.pl -g $testfile -s ${basen}.predictions.conll > ${basen}.test.eval.out 2> eval.err
echo "wrote to ${basen}.test.eval.out"
#scp ${basen}.test.eval.out pinot.cs.washington.edu:/m-pinotHD/pmulc/srl_outputs/results/${lang}/
