const sel = clip.selected.indices[0];
const co = clip.data['coords'];
const sam = clip.data['sample'];
const edge = clip.data['edges'];
const nscans = clip.data['nscans'];
const type = clip.data['type'];

function isDefined(x) {
    let undefined;
    return x !== undefined;
}

console.log(sel);
if(isDefined(sel)){
    co.splice(sel, 1);
    sam.splice(sel, 1);
    edge.splice(sel, 1);
    nscans.splice(sel, 1);
    type.splice(sel, 1);
    clip.change.emit();
}