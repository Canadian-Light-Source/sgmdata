var sel = clip.selected.indices[0];
var co = clip.data['coords'];
var sam = clip.data['sample'];
var edge = clip.data['edges'];
var nscans = clip.data['nscans'];
var type = clip.data['type'];

function isDefined(x) {
    var undefined;
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