var sdd = s2.data;
var aux_data = aux.data;
var filetext = 'Energy,I0,TEY,Diode,SDD\\n';

xstart = aux_data['en'].findIndex((x) => x >=sdd['en'][0])
for (i=0; i < sdd['en'].length; i++) {
    if(isNaN(sdd['proj_y'][i])){ continue; }
    else{
        var currRow = [aux_data['en'][i + xstart].toString(),aux_data['i0'][i+xstart].toString(), aux_data['tey'][i + xstart].toString(), aux_data['pd'][i + xstart].toString(), sdd['proj_y'][i].toString().concat('\\n')];
        var joined = currRow.join();
        filetext = filetext.concat(joined);
    }
}
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });
//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}else {
    var link = document.createElement("a");
    link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'));
}