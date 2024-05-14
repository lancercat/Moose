class init_layer_g2_bws:
    def __init__(this, layer_dict,bn_dict,layer_container,bn_container):
        this.conv=layer_container.model[layer_dict["conv"]];
        this.bn=bn_container.model[bn_dict["bn"]];
        this.relu=layer_container.model[layer_dict["relu"]];
    def __call__(this,x):
        a=this.conv(x);
        b=this.bn(a);
        return this.relu(b);

class BasicBlock_ass_g2_bws:
    def __init__(this,  layer_dict,bn_dict,layer_container,bn_container):
        this.conv1=layer_container.model[layer_dict["conv1"]];
        this.relu=layer_container.model[layer_dict["relu"]];
        this.bn1=bn_container.model[bn_dict["bn1"]];
        this.conv2=layer_container.model[layer_dict["conv2"]];
        this.bn2 = bn_container.model[bn_dict["bn2"]];
        this.downsample =False;
        if("downsample" in layer_dict):
            this.sample_conv=layer_container.model[layer_dict["downsample"]["conv"]];
            this.sample_bn=bn_container.model[bn_dict["downsample"]["bn"]];
            this.downsample = True;


    def __call__(this,x ):
        residual = x
        out = this.conv1(x)
        out = this.bn1(out)
        out = this.relu(out)

        out = this.conv2(out)
        out = this.bn2(out)

        if this.downsample:
            residual = this.sample_conv(x);
            residual = this.sample_bn(residual);
        out += residual
        out = this.relu(out)

        return out


class dan_reslayer_g2_bws:
    BLK=BasicBlock_ass_g2_bws
    def __init__(this,  layer_dict,bn_dict,layer_container,bn_container):
        this.blocks=[];
        for k in layer_dict["blocks"]:
            blk=this.BLK(layer_dict["blocks"][k],bn_dict["blocks"][k],layer_container,bn_container)
            this.blocks.append(blk);
    def __call__(this,x):
        for l in this.blocks:
            x=l(x);
        return x;