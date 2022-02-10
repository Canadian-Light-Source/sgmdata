var fluo = sl.value;
var idx = fluo / 51;
var peak = rect.data['x'];
var d = source.data['z'];
var f = det.value;
if (f.includes('sdd')){
    var sdd =  source.data[f + '-' + idx.toString()];
    for (i = 0; i < d.length; i++) {
        d[i] = sdd[i];
    }
}
if (f === "tey") {
    var tey = source.data['tey'];
    for (i = 0; i < d.length; i++) {
        d[i] = tey[i];
    }
}
peak[0] = fluo;
rect.change.emit();
source.change.emit();