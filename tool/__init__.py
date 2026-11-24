from .bridge import (
    DEFAULT_BIOMEDPARSE_URL,
    DEFAULT_MEDSAM3_URL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RETRIEVE_URL,
    DEFAULT_SANDBOX_URL,
    ToolBridge,
)
from .common import ToolRuntimeContext


def build_toolset(runtime: ToolRuntimeContext, enabled_tools=None):
    from .biomedparse_tool import BiomedParseTool
    from .jupyter_sandbox_tool import JupyterSandboxTool
    from .medsam3_tool import MedSAM3Tool
    from .retrieve_tool import RetrieveTool

    enabled = set(enabled_tools or ['biomedparse', 'retrieve', 'medsam3'])
    tools = []
    if 'biomedparse' in enabled:
        tools.append(BiomedParseTool(runtime=runtime))
    if 'retrieve' in enabled:
        tools.append(RetrieveTool())
    if 'medsam3' in enabled:
        tools.append(MedSAM3Tool(runtime=runtime))
    if 'jupyter_sandbox' in enabled or 'python_exec' in enabled or 'code_sandbox' in enabled:
        tools.append(JupyterSandboxTool(runtime=runtime))
    return tools
