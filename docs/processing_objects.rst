

Guidelines for processing objects
=================================


Class derivation
----------------

All processing objects derive from the :py:class:`~specula.base_processing_obj.BaseProcessingObj` class.

Life cycle
----------

* Init
* input/output connection
* Setup
* Loop (repeated N times):

  * prepare_trigger
  * trigger_code
  * post_trigger
* Finalize
 

Initialization
--------------

The *__init__* parameters should configure the object as much as possible. Default values are allowed. Use type hints whenever possible, either Python built-in types or SPECULA data objects. Parameters (of any type) that can be omitted completely should be initialized with None.

All processing objects must accept the *target_device_idx* and *precision* arguments, initialized to default values of None, and call the superclass *__init__* method with those two arguments::

    class ExampleProcessingObj(BaseProcessingObj):
        '''Example processing object'''

        def __init__(self,
                     foo: int,
                     bar: str=None,
                     target_device_idx=None,
                     precision=None
                ):
            super().__init__(target_device_idx=target_device_idx, precision=precision)



The *__init__* method must also allocate all inputs and outputs for this instance, using the built-in *self.inputs* and *self.outputs* dictionaries.

Inputs
******

Each input is an entry on the *self.inputs* dictionary, and is an instance of the :py:class:`~specula.connections.InputValue` class, initialized with the expected input type. It is possible for an input to be a list of values of the same type, in this case use an instance of :py:class:`~specula.connections.InputList`::

        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['in_slopes_list'] = InputList(type=Slopes)

If the input is optional, add *optional=True* (by default, all inputs are mandatory)::

        self.inputs['in_slopes'] = InputValue(type=Slopes, optional=True)

Calling the .get() method of an optional input may return None if it has not been set.

Outputs
*******

Outputs are instances of SPECULA data objects that must be allocated exactly once in the object constructor, and then never reassigned: rather, change their contents to refresh the output. The *self.outputs* dictionary must contain a reference to each output::

        self.modes = BaseValue('output modes from modal reconstructor', target_device_idx=target_device_idx)
        self.outputs['out_modes'] = self.modes

The output type is inferred automatically





