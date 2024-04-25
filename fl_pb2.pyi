from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelTypes(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    MODEL_TYPES_GLOBAL: _ClassVar[ModelTypes]
    MODEL_TYPES_MEC: _ClassVar[ModelTypes]
    MODEL_TYPES_USER: _ClassVar[ModelTypes]
MODEL_TYPES_GLOBAL: ModelTypes
MODEL_TYPES_MEC: ModelTypes
MODEL_TYPES_USER: ModelTypes

class Section(_message.Message):
    __slots__ = ["options"]
    class OptionsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    options: _containers.ScalarMap[str, str]
    def __init__(self, options: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Config(_message.Message):
    __slots__ = ["sections"]
    class SectionsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Section
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Section, _Mapping]] = ...) -> None: ...
    SECTIONS_FIELD_NUMBER: _ClassVar[int]
    sections: _containers.MessageMap[str, Section]
    def __init__(self, sections: _Optional[_Mapping[str, Section]] = ...) -> None: ...

class ReturnConfigMsg(_message.Message):
    __slots__ = ["ret_msg"]
    RET_MSG_FIELD_NUMBER: _ClassVar[int]
    ret_msg: str
    def __init__(self, ret_msg: _Optional[str] = ...) -> None: ...

class BasicMsg(_message.Message):
    __slots__ = ["none"]
    NONE_FIELD_NUMBER: _ClassVar[int]
    none: bool
    def __init__(self, none: bool = ...) -> None: ...

class FinalResultsMsg(_message.Message):
    __slots__ = ["g_round", "accuracy", "loss", "num_aggs", "elapsed_time"]
    G_ROUND_FIELD_NUMBER: _ClassVar[int]
    ACCURACY_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    NUM_AGGS_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_TIME_FIELD_NUMBER: _ClassVar[int]
    g_round: int
    accuracy: float
    loss: float
    num_aggs: int
    elapsed_time: float
    def __init__(self, g_round: _Optional[int] = ..., accuracy: _Optional[float] = ..., loss: _Optional[float] = ..., num_aggs: _Optional[int] = ..., elapsed_time: _Optional[float] = ...) -> None: ...

class ConditionMsg(_message.Message):
    __slots__ = ["id", "exp_name", "condition"]
    ID_FIELD_NUMBER: _ClassVar[int]
    EXP_NAME_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    id: str
    exp_name: str
    condition: bool
    def __init__(self, id: _Optional[str] = ..., exp_name: _Optional[str] = ..., condition: bool = ...) -> None: ...

class AddressMsg(_message.Message):
    __slots__ = ["id", "ip", "port"]
    ID_FIELD_NUMBER: _ClassVar[int]
    IP_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    id: str
    ip: str
    port: str
    def __init__(self, id: _Optional[str] = ..., ip: _Optional[str] = ..., port: _Optional[str] = ...) -> None: ...

class AddressSetMsg(_message.Message):
    __slots__ = ["address_set"]
    ADDRESS_SET_FIELD_NUMBER: _ClassVar[int]
    address_set: _containers.RepeatedCompositeFieldContainer[AddressMsg]
    def __init__(self, address_set: _Optional[_Iterable[_Union[AddressMsg, _Mapping]]] = ...) -> None: ...

class ModelMsg(_message.Message):
    __slots__ = ["sender_addr", "fl_round", "model_layers", "model_layers_shape", "num_trained_data", "num_trained_data_per_label", "num_epoch", "model_hash", "mec_index", "userset_index", "current_accuracy", "finish"]
    SENDER_ADDR_FIELD_NUMBER: _ClassVar[int]
    FL_ROUND_FIELD_NUMBER: _ClassVar[int]
    MODEL_LAYERS_FIELD_NUMBER: _ClassVar[int]
    MODEL_LAYERS_SHAPE_FIELD_NUMBER: _ClassVar[int]
    NUM_TRAINED_DATA_FIELD_NUMBER: _ClassVar[int]
    NUM_TRAINED_DATA_PER_LABEL_FIELD_NUMBER: _ClassVar[int]
    NUM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    MODEL_HASH_FIELD_NUMBER: _ClassVar[int]
    MEC_INDEX_FIELD_NUMBER: _ClassVar[int]
    USERSET_INDEX_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ACCURACY_FIELD_NUMBER: _ClassVar[int]
    FINISH_FIELD_NUMBER: _ClassVar[int]
    sender_addr: AddressMsg
    fl_round: int
    model_layers: _containers.RepeatedScalarFieldContainer[bytes]
    model_layers_shape: _containers.RepeatedScalarFieldContainer[str]
    num_trained_data: int
    num_trained_data_per_label: _containers.RepeatedScalarFieldContainer[int]
    num_epoch: _containers.RepeatedScalarFieldContainer[int]
    model_hash: str
    mec_index: int
    userset_index: int
    current_accuracy: float
    finish: bool
    def __init__(self, sender_addr: _Optional[_Union[AddressMsg, _Mapping]] = ..., fl_round: _Optional[int] = ..., model_layers: _Optional[_Iterable[bytes]] = ..., model_layers_shape: _Optional[_Iterable[str]] = ..., num_trained_data: _Optional[int] = ..., num_trained_data_per_label: _Optional[_Iterable[int]] = ..., num_epoch: _Optional[_Iterable[int]] = ..., model_hash: _Optional[str] = ..., mec_index: _Optional[int] = ..., userset_index: _Optional[int] = ..., current_accuracy: _Optional[float] = ..., finish: bool = ...) -> None: ...

class ModelSetMsg(_message.Message):
    __slots__ = ["model_set"]
    MODEL_SET_FIELD_NUMBER: _ClassVar[int]
    model_set: _containers.RepeatedCompositeFieldContainer[ModelMsg]
    def __init__(self, model_set: _Optional[_Iterable[_Union[ModelMsg, _Mapping]]] = ...) -> None: ...

class ReqUserModelSetMsg(_message.Message):
    __slots__ = ["model", "address_set", "foo"]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_SET_FIELD_NUMBER: _ClassVar[int]
    FOO_FIELD_NUMBER: _ClassVar[int]
    model: ModelMsg
    address_set: _containers.RepeatedCompositeFieldContainer[AddressMsg]
    foo: int
    def __init__(self, model: _Optional[_Union[ModelMsg, _Mapping]] = ..., address_set: _Optional[_Iterable[_Union[AddressMsg, _Mapping]]] = ..., foo: _Optional[int] = ...) -> None: ...

class ReturnMsg(_message.Message):
    __slots__ = ["accepted", "sender_addr", "fl_round", "num_trained_data", "current_accuracy"]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    SENDER_ADDR_FIELD_NUMBER: _ClassVar[int]
    FL_ROUND_FIELD_NUMBER: _ClassVar[int]
    NUM_TRAINED_DATA_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ACCURACY_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    sender_addr: AddressMsg
    fl_round: int
    num_trained_data: int
    current_accuracy: float
    def __init__(self, accepted: bool = ..., sender_addr: _Optional[_Union[AddressMsg, _Mapping]] = ..., fl_round: _Optional[int] = ..., num_trained_data: _Optional[int] = ..., current_accuracy: _Optional[float] = ...) -> None: ...

class ReturnSetMsg(_message.Message):
    __slots__ = ["return_set"]
    RETURN_SET_FIELD_NUMBER: _ClassVar[int]
    return_set: _containers.RepeatedCompositeFieldContainer[ReturnMsg]
    def __init__(self, return_set: _Optional[_Iterable[_Union[ReturnMsg, _Mapping]]] = ...) -> None: ...
