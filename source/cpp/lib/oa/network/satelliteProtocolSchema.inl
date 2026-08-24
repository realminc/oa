// OA satellite protocol-v1 schema manifest. This file is included with
// OA_SATELLITE_MESSAGE and OA_SATELLITE_FIELD defined by the codec. It is the
// single owner of wire message/field identity and validation bounds.

OA_SATELLITE_MESSAGE(Hello, 1)
OA_SATELLITE_FIELD(Hello, HelloProtocolMin, 1, U16, true, 2, 2)
OA_SATELLITE_FIELD(Hello, HelloProtocolMax, 2, U16, true, 2, 2)
OA_SATELLITE_FIELD(Hello, HelloClientNonce, 3, Bytes, true, 32, 32)
OA_SATELLITE_FIELD(Hello, HelloAuthProof, 4, Bytes, true, 32, 32)
OA_SATELLITE_FIELD(Hello, HelloBuildHash, 5, Bytes, true, 32, 32)
OA_SATELLITE_FIELD(Hello, HelloSchemaHash, 6, Bytes, true, 32, 32)
OA_SATELLITE_FIELD(Hello, HelloMaxPayloadBytes, 7, U32, true, 4, 4)
OA_SATELLITE_FIELD(Hello, HelloMaxResidentBytes, 8, U64, true, 8, 8)
OA_SATELLITE_FIELD(Hello, HelloMaxObjects, 9, U32, true, 4, 4)
OA_SATELLITE_FIELD(Hello, HelloMaxInflight, 10, U32, true, 4, 4)

OA_SATELLITE_MESSAGE(HelloReply, 2)
OA_SATELLITE_FIELD(HelloReply, HelloReplyProtocol, 1, U16, true, 2, 2)
OA_SATELLITE_FIELD(HelloReply, HelloReplyServerNonce, 2, Bytes, true, 32, 32)
OA_SATELLITE_FIELD(HelloReply, HelloReplyAuthProof, 3, Bytes, true, 32, 32)
OA_SATELLITE_FIELD(HelloReply, HelloReplySessionEpoch, 4, U64, true, 8, 8)
OA_SATELLITE_FIELD(HelloReply, HelloReplyDeviceName, 5, String, true, 1, 256)
OA_SATELLITE_FIELD(HelloReply, HelloReplyMaxPayloadBytes, 6, U32, true, 4, 4)
OA_SATELLITE_FIELD(HelloReply, HelloReplyMaxResidentBytes, 7, U64, true, 8, 8)
OA_SATELLITE_FIELD(HelloReply, HelloReplyMaxObjects, 8, U32, true, 4, 4)
OA_SATELLITE_FIELD(HelloReply, HelloReplyMaxInflight, 9, U32, true, 4, 4)

OA_SATELLITE_MESSAGE(PutObject, 3)
OA_SATELLITE_FIELD(PutObject, PutObjectId, 1, U64, true, 8, 8)
OA_SATELLITE_FIELD(PutObject, PutObjectDtype, 2, U8, true, 1, 1)
OA_SATELLITE_FIELD(PutObject, PutObjectShape, 3, I64Array, true, 8, 64)
OA_SATELLITE_FIELD(PutObject, PutObjectData, 4, Bytes, true, 0, 1048576)
OA_SATELLITE_FIELD(PutObject, PutObjectContentHash, 5, Bytes, true, 32, 32)
OA_SATELLITE_FIELD(PutObject, PutObjectVersion, 6, U64, true, 8, 8)

OA_SATELLITE_MESSAGE(dropObject, 4)
OA_SATELLITE_FIELD(dropObject, DropObjectId, 1, U64, true, 8, 8)

OA_SATELLITE_MESSAGE(ExecuteNamed, 5)
OA_SATELLITE_FIELD(ExecuteNamed, ExecuteOperation, 1, String, true, 1, 128)
OA_SATELLITE_FIELD(ExecuteNamed, ExecuteInputObjectIds, 2, U64Array, true, 0, 512)
OA_SATELLITE_FIELD(ExecuteNamed, ExecuteOutputObjectId, 3, U64, true, 8, 8)
OA_SATELLITE_FIELD(ExecuteNamed, ExecuteArguments, 4, Bytes, false, 0, 4096)
OA_SATELLITE_FIELD(ExecuteNamed, ExecuteExpectedVersion, 5, U64, false, 8, 8)
OA_SATELLITE_FIELD(ExecuteNamed, ExecuteExpectedHash, 6, Bytes, false, 32, 32)

OA_SATELLITE_MESSAGE(wait, 6)
OA_SATELLITE_FIELD(wait, WaitRequestId, 1, U64, true, 8, 8)

OA_SATELLITE_MESSAGE(poll, 7)
OA_SATELLITE_FIELD(poll, PollRequestId, 1, U64, true, 8, 8)

OA_SATELLITE_MESSAGE(cancel, 8)
OA_SATELLITE_FIELD(cancel, CancelRequestId, 1, U64, true, 8, 8)

OA_SATELLITE_MESSAGE(getResult, 9)
OA_SATELLITE_FIELD(getResult, GetResultRequestId, 1, U64, true, 8, 8)

OA_SATELLITE_MESSAGE(Error, 10)
OA_SATELLITE_FIELD(Error, ErrorStatusCode, 1, U32, true, 4, 4)
OA_SATELLITE_FIELD(Error, ErrorMessage, 2, String, true, 1, 1024)
OA_SATELLITE_FIELD(Error, ErrorPoisoned, 3, U8, true, 1, 1)

OA_SATELLITE_MESSAGE(abort, 11)
OA_SATELLITE_FIELD(abort, AbortReason, 1, String, true, 1, 1024)

OA_SATELLITE_MESSAGE(Close, 12)

OA_SATELLITE_MESSAGE(result, 13)
OA_SATELLITE_FIELD(result, ResultRequestId, 1, U64, true, 8, 8)
OA_SATELLITE_FIELD(result, ResultComplete, 2, U8, true, 1, 1)
OA_SATELLITE_FIELD(result, ResultStatusCode, 3, U32, true, 4, 4)
OA_SATELLITE_FIELD(result, ResultBytes, 4, Bytes, false, 0, 1048576)
OA_SATELLITE_FIELD(result, ResultProfile, 5, Bytes, false, 0, 4096)
OA_SATELLITE_FIELD(result, ResultObjectId, 6, U64, false, 8, 8)
OA_SATELLITE_FIELD(result, ResultObjectVersion, 7, U64, false, 8, 8)
OA_SATELLITE_FIELD(result, ResultObjectHash, 8, Bytes, false, 32, 32)
