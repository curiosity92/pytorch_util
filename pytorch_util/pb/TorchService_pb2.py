# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: TorchService.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='TorchService.proto',
  package='PyTorchService',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12TorchService.proto\x12\x0ePyTorchService\"\'\n\x14\x45xtractSingleRequest\x12\x0f\n\x07\x62ytes_x\x18\x01 \x01(\x0c\"0\n#ExtractSingleClassificationResponse\x12\t\n\x01y\x18\x01 \x01(\x05\"+\n\x13\x45xtractBatchRequest\x12\x14\n\x0c\x62ytes_list_x\x18\x01 \x01(\x0c\",\n\x14\x45xtractBatchResponse\x12\x14\n\x0c\x62ytes_list_y\x18\x01 \x01(\x0c\x32\xdb\x01\n\x0cTorchService\x12\\\n\rextract_batch\x12#.PyTorchService.ExtractBatchRequest\x1a$.PyTorchService.ExtractBatchResponse\"\x00\x12m\n\x0e\x65xtract_single\x12$.PyTorchService.ExtractSingleRequest\x1a\x33.PyTorchService.ExtractSingleClassificationResponse\"\x00\x62\x06proto3'
)




_EXTRACTSINGLEREQUEST = _descriptor.Descriptor(
  name='ExtractSingleRequest',
  full_name='PyTorchService.ExtractSingleRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='bytes_x', full_name='PyTorchService.ExtractSingleRequest.bytes_x', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=38,
  serialized_end=77,
)


_EXTRACTSINGLECLASSIFICATIONRESPONSE = _descriptor.Descriptor(
  name='ExtractSingleClassificationResponse',
  full_name='PyTorchService.ExtractSingleClassificationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='y', full_name='PyTorchService.ExtractSingleClassificationResponse.y', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=79,
  serialized_end=127,
)


_EXTRACTBATCHREQUEST = _descriptor.Descriptor(
  name='ExtractBatchRequest',
  full_name='PyTorchService.ExtractBatchRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='bytes_list_x', full_name='PyTorchService.ExtractBatchRequest.bytes_list_x', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=129,
  serialized_end=172,
)


_EXTRACTBATCHRESPONSE = _descriptor.Descriptor(
  name='ExtractBatchResponse',
  full_name='PyTorchService.ExtractBatchResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='bytes_list_y', full_name='PyTorchService.ExtractBatchResponse.bytes_list_y', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=174,
  serialized_end=218,
)

DESCRIPTOR.message_types_by_name['ExtractSingleRequest'] = _EXTRACTSINGLEREQUEST
DESCRIPTOR.message_types_by_name['ExtractSingleClassificationResponse'] = _EXTRACTSINGLECLASSIFICATIONRESPONSE
DESCRIPTOR.message_types_by_name['ExtractBatchRequest'] = _EXTRACTBATCHREQUEST
DESCRIPTOR.message_types_by_name['ExtractBatchResponse'] = _EXTRACTBATCHRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ExtractSingleRequest = _reflection.GeneratedProtocolMessageType('ExtractSingleRequest', (_message.Message,), {
  'DESCRIPTOR' : _EXTRACTSINGLEREQUEST,
  '__module__' : 'TorchService_pb2'
  # @@protoc_insertion_point(class_scope:PyTorchService.ExtractSingleRequest)
  })
_sym_db.RegisterMessage(ExtractSingleRequest)

ExtractSingleClassificationResponse = _reflection.GeneratedProtocolMessageType('ExtractSingleClassificationResponse', (_message.Message,), {
  'DESCRIPTOR' : _EXTRACTSINGLECLASSIFICATIONRESPONSE,
  '__module__' : 'TorchService_pb2'
  # @@protoc_insertion_point(class_scope:PyTorchService.ExtractSingleClassificationResponse)
  })
_sym_db.RegisterMessage(ExtractSingleClassificationResponse)

ExtractBatchRequest = _reflection.GeneratedProtocolMessageType('ExtractBatchRequest', (_message.Message,), {
  'DESCRIPTOR' : _EXTRACTBATCHREQUEST,
  '__module__' : 'TorchService_pb2'
  # @@protoc_insertion_point(class_scope:PyTorchService.ExtractBatchRequest)
  })
_sym_db.RegisterMessage(ExtractBatchRequest)

ExtractBatchResponse = _reflection.GeneratedProtocolMessageType('ExtractBatchResponse', (_message.Message,), {
  'DESCRIPTOR' : _EXTRACTBATCHRESPONSE,
  '__module__' : 'TorchService_pb2'
  # @@protoc_insertion_point(class_scope:PyTorchService.ExtractBatchResponse)
  })
_sym_db.RegisterMessage(ExtractBatchResponse)



_TORCHSERVICE = _descriptor.ServiceDescriptor(
  name='TorchService',
  full_name='PyTorchService.TorchService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=221,
  serialized_end=440,
  methods=[
  _descriptor.MethodDescriptor(
    name='extract_batch',
    full_name='PyTorchService.TorchService.extract_batch',
    index=0,
    containing_service=None,
    input_type=_EXTRACTBATCHREQUEST,
    output_type=_EXTRACTBATCHRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='extract_single',
    full_name='PyTorchService.TorchService.extract_single',
    index=1,
    containing_service=None,
    input_type=_EXTRACTSINGLEREQUEST,
    output_type=_EXTRACTSINGLECLASSIFICATIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_TORCHSERVICE)

DESCRIPTOR.services_by_name['TorchService'] = _TORCHSERVICE

# @@protoc_insertion_point(module_scope)
