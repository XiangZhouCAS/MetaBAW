mkdir -p tmp_workfolder
cat test.fa | \
parallel -j 200 --block 999k --recstart '>' --pipe \
"prodigal -p meta -a tmp_workfolder/tmp_{#}.faa -d tmp_workfolder/tmp_{#}.ffn -o tmp_workfolder/tmp_{#}.gff"
cat tmp_workfolder/*.faa > test.faa
rm -r tmp_workfolder
