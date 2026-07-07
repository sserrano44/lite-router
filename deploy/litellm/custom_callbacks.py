"""Entry point referenced by litellm_config.yaml:

    litellm_settings:
      callbacks: custom_callbacks.proxy_handler_instance
"""

from session_router.hook import RipioAutoRouter

proxy_handler_instance = RipioAutoRouter()
