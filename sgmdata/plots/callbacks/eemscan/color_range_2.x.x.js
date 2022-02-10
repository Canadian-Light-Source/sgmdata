const o_min = cb_obj.value[0];
const o_max = cb_obj.value[1];
if (typeof im.glyph.color_mapper !== 'undefined'){
    im.glyph.color_mapper.low = o_min;
}else{
    im.glyph.fill_color.transform.low = o_min;
}
if (typeof im.glyph.color_mapper !== 'undefined'){
    im.glyph.color_mapper.high = o_max;
}else{
    im.glyph.fill_color.transform.high = o_max;
}
cl.color_mapper.low = o_min;
cl.color_mapper.high = o_max;
