..
    The empty line below should not be removed. It is added such that the `rst_prolog`
    is added before the :mod: directive. Otherwise, the rendering will show as a
    paragraph instead of a header.

:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

{% if is_pydantic_model(module, objname) %}
.. autopydantic_model:: {{ objname }}
    :members:
{% else %}
.. autoclass:: {{ objname }}
    :members:
{% endif %}

.. raw:: html

    <div class="clearer"></div>
