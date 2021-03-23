import pytest
import magma
from magma import clear_cachedFunctions
from gemstone.generator import clear_generator_cache


@pytest.fixture(autouse=True)
def magma_test():
    clear_cachedFunctions()
    magma.frontend.coreir_.ResetCoreIR()
    clear_generator_cache()
