let text = tx.value;
const samples = clip.data['sample'];
const edges = clip.data['edges'];
const nscans = clip.data['nscans'];
const pos = clip.data['coords'];
const type = clip.data['type'];
let scan;
let ncols;
let typenum = 1;
let i;

text = "plate = [\n";
for (i = 0; i < pos.length; i++) {
    scan = lib[edges[i]]
    if (type[i] === "reference"){
        typenum = 2;
    }
    else{
        typenum = 1;
    }
    if (edges[i] === "EEMs"){
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
