const Viridis256  = [0x440154ff, 0x440255ff, 0x440357ff, 0x450558ff, 0x45065aff, 0x45085bff, 0x46095cff, 0x460b5eff, 0x460c5fff, 0x460e61ff, 0x470f62ff, 0x471163ff,
                0x471265ff, 0x471466ff, 0x471567ff, 0x471669ff, 0x47186aff, 0x48196bff, 0x481a6cff, 0x481c6eff, 0x481d6fff, 0x481e70ff, 0x482071ff, 0x482172ff,
                0x482273ff, 0x482374ff, 0x472575ff, 0x472676ff, 0x472777ff, 0x472878ff, 0x472a79ff, 0x472b7aff, 0x472c7bff, 0x462d7cff, 0x462f7cff, 0x46307dff,
                0x46317eff, 0x45327fff, 0x45347fff, 0x453580ff, 0x453681ff, 0x443781ff, 0x443982ff, 0x433a83ff, 0x433b83ff, 0x433c84ff, 0x423d84ff, 0x423e85ff,
                0x424085ff, 0x414186ff, 0x414286ff, 0x404387ff, 0x404487ff, 0x3f4587ff, 0x3f4788ff, 0x3e4888ff, 0x3e4989ff, 0x3d4a89ff, 0x3d4b89ff, 0x3d4c89ff,
                0x3c4d8aff, 0x3c4e8aff, 0x3b508aff, 0x3b518aff, 0x3a528bff, 0x3a538bff, 0x39548bff, 0x39558bff, 0x38568bff, 0x38578cff, 0x37588cff, 0x37598cff,
                0x365a8cff, 0x365b8cff, 0x355c8cff, 0x355d8cff, 0x345e8dff, 0x345f8dff, 0x33608dff, 0x33618dff, 0x32628dff, 0x32638dff, 0x31648dff, 0x31658dff,
                0x31668dff, 0x30678dff, 0x30688dff, 0x2f698dff, 0x2f6a8dff, 0x2e6b8eff, 0x2e6c8eff, 0x2e6d8eff, 0x2d6e8eff, 0x2d6f8eff, 0x2c708eff, 0x2c718eff,
                0x2c728eff, 0x2b738eff, 0x2b748eff, 0x2a758eff, 0x2a768eff, 0x2a778eff, 0x29788eff, 0x29798eff, 0x287a8eff, 0x287a8eff, 0x287b8eff, 0x277c8eff,
                0x277d8eff, 0x277e8eff, 0x267f8eff, 0x26808eff, 0x26818eff, 0x25828eff, 0x25838dff, 0x24848dff, 0x24858dff, 0x24868dff, 0x23878dff, 0x23888dff,
                0x23898dff, 0x22898dff, 0x228a8dff, 0x228b8dff, 0x218c8dff, 0x218d8cff, 0x218e8cff, 0x208f8cff, 0x20908cff, 0x20918cff, 0x1f928cff, 0x1f938bff,
                0x1f948bff, 0x1f958bff, 0x1f968bff, 0x1e978aff, 0x1e988aff, 0x1e998aff, 0x1e998aff, 0x1e9a89ff, 0x1e9b89ff, 0x1e9c89ff, 0x1e9d88ff, 0x1e9e88ff,
                0x1e9f88ff, 0x1ea087ff, 0x1fa187ff, 0x1fa286ff, 0x1fa386ff, 0x20a485ff, 0x20a585ff, 0x21a685ff, 0x21a784ff, 0x22a784ff, 0x23a883ff, 0x23a982ff,
                0x24aa82ff, 0x25ab81ff, 0x26ac81ff, 0x27ad80ff, 0x28ae7fff, 0x29af7fff, 0x2ab07eff, 0x2bb17dff, 0x2cb17dff, 0x2eb27cff, 0x2fb37bff, 0x30b47aff,
                0x32b57aff, 0x33b679ff, 0x35b778ff, 0x36b877ff, 0x38b976ff, 0x39b976ff, 0x3bba75ff, 0x3dbb74ff, 0x3ebc73ff, 0x40bd72ff, 0x42be71ff, 0x44be70ff,
                0x45bf6fff, 0x47c06eff, 0x49c16dff, 0x4bc26cff, 0x4dc26bff, 0x4fc369ff, 0x51c468ff, 0x53c567ff, 0x55c666ff, 0x57c665ff, 0x59c764ff, 0x5bc862ff,
                0x5ec961ff, 0x60c960ff, 0x62ca5fff, 0x64cb5dff, 0x67cc5cff, 0x69cc5bff, 0x6bcd59ff, 0x6dce58ff, 0x70ce56ff, 0x72cf55ff, 0x74d054ff, 0x77d052ff,
                0x79d151ff, 0x7cd24fff, 0x7ed24eff, 0x81d34cff, 0x83d34bff, 0x86d449ff, 0x88d547ff, 0x8bd546ff, 0x8dd644ff, 0x90d643ff, 0x92d741ff, 0x95d73fff,
                0x97d83eff, 0x9ad83cff, 0x9dd93aff, 0x9fd938ff, 0xa2da37ff, 0xa5da35ff, 0xa7db33ff, 0xaadb32ff, 0xaddc30ff, 0xafdc2eff, 0xb2dd2cff, 0xb5dd2bff,
                0xb7dd29ff, 0xbade27ff, 0xbdde26ff, 0xbfdf24ff, 0xc2df22ff, 0xc5df21ff, 0xc7e01fff, 0xcae01eff, 0xcde01dff, 0xcfe11cff, 0xd2e11bff, 0xd4e11aff,
                0xd7e219ff, 0xdae218ff, 0xdce218ff, 0xdfe318ff, 0xe1e318ff, 0xe4e318ff, 0xe7e419ff, 0xe9e419ff, 0xece41aff, 0xeee51bff, 0xf1e51cff, 0xf3e51eff,
                0xf6e61fff, 0xf8e621ff, 0xfae622ff, 0xfde724ff]
const Spectral = [0x5e4fa2ff, 0x3288bdff, 0x66c2a5ff, 0xabdda4ff, 0xe6f598ff, 0xffffbfff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd53e4fff, 0x9e0142ff]
const Inferno256 = [0x000003ff, 0x000004ff, 0x000006ff, 0x010007ff, 0x010109ff, 0x01010bff, 0x02010eff, 0x020210ff, 0x030212ff, 0x040314ff, 0x040316ff, 0x050418ff,
                0x06041bff, 0x07051dff, 0x08061fff, 0x090621ff, 0x0a0723ff, 0x0b0726ff, 0x0d0828ff, 0x0e082aff, 0x0f092dff, 0x10092fff, 0x120a32ff, 0x130a34ff,
                0x140b36ff, 0x160b39ff, 0x170b3bff, 0x190b3eff, 0x1a0b40ff, 0x1c0c43ff, 0x1d0c45ff, 0x1f0c47ff, 0x200c4aff, 0x220b4cff, 0x240b4eff, 0x260b50ff,
                0x270b52ff, 0x290b54ff, 0x2b0a56ff, 0x2d0a58ff, 0x2e0a5aff, 0x300a5cff, 0x32095dff, 0x34095fff, 0x350960ff, 0x370961ff, 0x390962ff, 0x3b0964ff,
                0x3c0965ff, 0x3e0966ff, 0x400966ff, 0x410967ff, 0x430a68ff, 0x450a69ff, 0x460a69ff, 0x480b6aff, 0x4a0b6aff, 0x4b0c6bff, 0x4d0c6bff, 0x4f0d6cff,
                0x500d6cff, 0x520e6cff, 0x530e6dff, 0x550f6dff, 0x570f6dff, 0x58106dff, 0x5a116dff, 0x5b116eff, 0x5d126eff, 0x5f126eff, 0x60136eff, 0x62146eff,
                0x63146eff, 0x65156eff, 0x66156eff, 0x68166eff, 0x6a176eff, 0x6b176eff, 0x6d186eff, 0x6e186eff, 0x70196eff, 0x72196dff, 0x731a6dff, 0x751b6dff,
                0x761b6dff, 0x781c6dff, 0x7a1c6dff, 0x7b1d6cff, 0x7d1d6cff, 0x7e1e6cff, 0x801f6bff, 0x811f6bff, 0x83206bff, 0x85206aff, 0x86216aff, 0x88216aff,
                0x892269ff, 0x8b2269ff, 0x8d2369ff, 0x8e2468ff, 0x902468ff, 0x912567ff, 0x932567ff, 0x952666ff, 0x962666ff, 0x982765ff, 0x992864ff, 0x9b2864ff,
                0x9c2963ff, 0x9e2963ff, 0xa02a62ff, 0xa12b61ff, 0xa32b61ff, 0xa42c60ff, 0xa62c5fff, 0xa72d5fff, 0xa92e5eff, 0xab2e5dff, 0xac2f5cff, 0xae305bff,
                0xaf315bff, 0xb1315aff, 0xb23259ff, 0xb43358ff, 0xb53357ff, 0xb73456ff, 0xb83556ff, 0xba3655ff, 0xbb3754ff, 0xbd3753ff, 0xbe3852ff, 0xbf3951ff,
                0xc13a50ff, 0xc23b4fff, 0xc43c4eff, 0xc53d4dff, 0xc73e4cff, 0xc83e4bff, 0xc93f4aff, 0xcb4049ff, 0xcc4148ff, 0xcd4247ff, 0xcf4446ff, 0xd04544ff,
                0xd14643ff, 0xd24742ff, 0xd44841ff, 0xd54940ff, 0xd64a3fff, 0xd74b3eff, 0xd94d3dff, 0xda4e3bff, 0xdb4f3aff, 0xdc5039ff, 0xdd5238ff, 0xde5337ff,
                0xdf5436ff, 0xe05634ff, 0xe25733ff, 0xe35832ff, 0xe45a31ff, 0xe55b30ff, 0xe65c2eff, 0xe65e2dff, 0xe75f2cff, 0xe8612bff, 0xe9622aff, 0xea6428ff,
                0xeb6527ff, 0xec6726ff, 0xed6825ff, 0xed6a23ff, 0xee6c22ff, 0xef6d21ff, 0xf06f1fff, 0xf0701eff, 0xf1721dff, 0xf2741cff, 0xf2751aff, 0xf37719ff,
                0xf37918ff, 0xf47a16ff, 0xf57c15ff, 0xf57e14ff, 0xf68012ff, 0xf68111ff, 0xf78310ff, 0xf7850eff, 0xf8870dff, 0xf8880cff, 0xf88a0bff, 0xf98c09ff,
                0xf98e08ff, 0xf99008ff, 0xfa9107ff, 0xfa9306ff, 0xfa9506ff, 0xfa9706ff, 0xfb9906ff, 0xfb9b06ff, 0xfb9d06ff, 0xfb9e07ff, 0xfba007ff, 0xfba208ff,
                0xfba40aff, 0xfba60bff, 0xfba80dff, 0xfbaa0eff, 0xfbac10ff, 0xfbae12ff, 0xfbb014ff, 0xfbb116ff, 0xfbb318ff, 0xfbb51aff, 0xfbb71cff, 0xfbb91eff,
                0xfabb21ff, 0xfabd23ff, 0xfabf25ff, 0xfac128ff, 0xf9c32aff, 0xf9c52cff, 0xf9c72fff, 0xf8c931ff, 0xf8cb34ff, 0xf8cd37ff, 0xf7cf3aff, 0xf7d13cff,
                0xf6d33fff, 0xf6d542ff, 0xf5d745ff, 0xf5d948ff, 0xf4db4bff, 0xf4dc4fff, 0xf3de52ff, 0xf3e056ff, 0xf3e259ff, 0xf2e45dff, 0xf2e660ff, 0xf1e864ff,
                0xf1e968ff, 0xf1eb6cff, 0xf1ed70ff, 0xf1ee74ff, 0xf1f079ff, 0xf1f27dff, 0xf2f381ff, 0xf2f485ff, 0xf3f689ff, 0xf4f78dff, 0xf5f891ff, 0xf6fa95ff,
                0xf7fb99ff, 0xf9fc9dff, 0xfafda0ff, 0xfcfea4ff]
const f = cb_obj.value;
if (f === "Viridis") {
    if (typeof im.glyph.color_mapper !== 'undefined'){
        im.glyph.color_mapper.palette = Viridis256;
    }else{
        im.glyph.fill_color.transform.palette = Viridis256;
    }
    cl.color_mapper.palette = Viridis256;
}
if (f === "Spectral") {
    if (typeof im.glyph.color_mapper !== 'undefined'){
        im.glyph.color_mapper.palette = Spectral;
    }else{
        im.glyph.fill_color.transform.palette = Spectral;
    }
    cl.color_mapper.palette = Spectral;
}
if (f === "Inferno") {
    if (typeof im.glyph.color_mapper !== 'undefined'){
        im.glyph.color_mapper.palette = Inferno256;
    }else{
        im.glyph.fill_color.transform.palette = Inferno256;
    }
    cl.color_mapper.palette = Inferno256;
}