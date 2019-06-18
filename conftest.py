import pytest
from magma import clear_cachedFunctions
import magma.backend.coreir_ as coreir_
from gemstone.generator import clear_generator_cache


@pytest.fixture(autouse=True)
def magma_test():
    clear_cachedFunctions()
    coreir_.CoreIRContextSingleton().reset_instance()
    clear_generator_cache()
