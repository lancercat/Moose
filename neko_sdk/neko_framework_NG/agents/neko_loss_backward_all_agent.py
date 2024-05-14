from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_sync_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace,neko_environment
# which sums all losses in objdict and commence backward function.

class neko_basic_backward_all_agent(neko_abstract_sync_agent):
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        tl=0;
        for l in workspace.objdict:
            tl=tl+workspace.objdict[l];
        tl.backward();
        # logging is nomore managed by backward agent,
        workspace.objdict={};
        return workspace;

def get_neko_basic_backward_all_agent():
    return {
        "agent":neko_basic_backward_all_agent,
        "params":{}
    }