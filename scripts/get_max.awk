BEGIN { print "\nFName","\t","Max"; print "-----------------"; max=0 } 
$2 > max { max=$2;f=$1 } 
END { print f,"\t",max; print "" }
