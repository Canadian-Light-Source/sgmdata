const sdd1 = source.data['im1'][0];
const sdd2 = source.data['im2'][0];
const sdd3 = source.data['im3'][0];
const sdd4 = source.data['im4'][0];
const tey = source.data['im5'][0];
let d = source.data['image'];
const f = cb_obj.value;
if (f === "sdd1") {
    d[0] = sdd1;
}
if (f === "sdd2") {
    d[0] = sdd2;
}
if (f === "sdd3") {
    d[0] = sdd3;
}
if (f === "sdd4") {
    d[0] = sdd4;
}
if (f === "tey") {
    d[0] = tey;
}
source.change.emit();