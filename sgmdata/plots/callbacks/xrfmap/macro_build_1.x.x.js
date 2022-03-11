var text = tx.value;
var samples = clip.data['sample'];
var edges = clip.data['edges'];
var nscans = clip.data['nscans'];
var pos = clip.data['coords'];
var type = clip.data['type'];
var scan;
var ncols;
var typenum = 1;
var i;

text = "plate = [\n";
for (i = 0; i < pos.length; i++) {
    scan = lib[edges[i]]
    if (type[i] == "reference"){
        typenum = 2;
    }
    else{
        typenum = 1;
    }
    if (edges[i] == "EEMs"){
        text += "{'sample': '" + samples[i] + "'";
        text += ", 'type': " + typenum.toString();
        text += ", 'scan': '" + scan + "'";
        text += ", 'coords': " + pos[i] + "},\n";
    }
    else{
        ncols = Math.ceil(nscans[i] / 10);
        scan = scan.replace('col', ncols.toString());
        text += "{'sample': '" + samples[i] + " - " + edges[i] + "'";
        text += ", 'type': " + typenum.toString();
        text += ", 'scan': '" + scan + "'";
        text += ", 'coords': " + pos[i] + "},\n";
    }
}
text += "]";
tx.value = text;
