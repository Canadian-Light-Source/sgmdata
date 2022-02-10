const fluo = sl.value;
const idx = fluo / 51;
const peak = rect.data['x'];
let d = source.data['z'];
const f = det.value;
let i;
if (f.includes('sdd')){
    let sdd =  source.data[f + '-' + idx.toString()];
    for (i = 0; i < d.length; i++) {
        d[i] = sdd[i];
    }
}
if (f === "tey") {
    let tey = source.data['tey'];
    for (i = 0; i < d.length; i++) {
        d[i] = tey[i];
    }
}
peak[0] = fluo;
rect.change.emit();
source.change.emit();