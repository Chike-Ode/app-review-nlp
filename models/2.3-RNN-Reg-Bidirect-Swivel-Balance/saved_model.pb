??;
?,?,
:
Add
x"T
y"T
z"T"
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
?
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0?????????"
value_indexint(0?????????"+

vocab_sizeint?????????(0?????????"
	delimiterstring	"
offsetint ?
:
Less
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
2
LookupTableSizeV2
table_handle
size	?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
?
SparseSegmentSqrtN	
data"T
indices"Tidx
segment_ids"Tsegmentids
output"T"
Ttype:
2"
Tidxtype0:
2	"
Tsegmentidstype0:
2	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
StringSplit	
input
	delimiter
indices	

values	
shape	"

skip_emptybool(
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?
E
Where

input"T	
index	"%
Ttype0
:
2	

?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.6.22unknown8??9
n
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name
Variable
g
Variable/Read/ReadVariableOpReadVariableOpVariable* 
_output_shapes
:
??*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
9bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*J
shared_name;9bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel
?
Mbidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/Read/ReadVariableOpReadVariableOp9bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel*
_output_shapes

:
*
dtype0
?
Cbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*T
shared_nameECbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel
?
Wbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpCbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel*
_output_shapes

:

*
dtype0
?
7bidirectional/forward_simple_rnn/simple_rnn_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*H
shared_name97bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias
?
Kbidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/Read/ReadVariableOpReadVariableOp7bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias*
_output_shapes
:
*
dtype0
?
:bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*K
shared_name<:bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel
?
Nbidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/Read/ReadVariableOpReadVariableOp:bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel*
_output_shapes

:
*
dtype0
?
Dbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*U
shared_nameFDbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel
?
Xbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOpDbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel*
_output_shapes

:

*
dtype0
?
8bidirectional/backward_simple_rnn/simple_rnn_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*I
shared_name:8bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias
?
Lbidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/Read/ReadVariableOpReadVariableOp8bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias*
_output_shapes
:
*
dtype0
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
V
Variable_1/AssignAssignVariableOp
Variable_1asset_path_initializer*
dtype0
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_383840
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
|
Adam/Variable/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_nameAdam/Variable/m
u
#Adam/Variable/m/Read/ReadVariableOpReadVariableOpAdam/Variable/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*Q
shared_nameB@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/m
?
TAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/m*
_output_shapes

:
*
dtype0
?
JAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*[
shared_nameLJAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/m
?
^Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpJAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/m*
_output_shapes

:

*
dtype0
?
>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*O
shared_name@>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/m
?
RAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/m/Read/ReadVariableOpReadVariableOp>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/m*
_output_shapes
:
*
dtype0
?
AAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*R
shared_nameCAAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/m
?
UAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/m*
_output_shapes

:
*
dtype0
?
KAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*\
shared_nameMKAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/m
?
_Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpKAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/m*
_output_shapes

:

*
dtype0
?
?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*P
shared_nameA?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/m
?
SAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/m/Read/ReadVariableOpReadVariableOp?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/m*
_output_shapes
:
*
dtype0
|
Adam/Variable/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_nameAdam/Variable/v
u
#Adam/Variable/v/Read/ReadVariableOpReadVariableOpAdam/Variable/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*Q
shared_nameB@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/v
?
TAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/v*
_output_shapes

:
*
dtype0
?
JAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*[
shared_nameLJAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/v
?
^Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpJAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/v*
_output_shapes

:

*
dtype0
?
>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*O
shared_name@>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/v
?
RAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/v/Read/ReadVariableOpReadVariableOp>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/v*
_output_shapes
:
*
dtype0
?
AAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*R
shared_nameCAAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/v
?
UAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/v*
_output_shapes

:
*
dtype0
?
KAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*\
shared_nameMKAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/v
?
_Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpKAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/v*
_output_shapes

:

*
dtype0
?
?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*P
shared_nameA?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/v
?
SAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/v/Read/ReadVariableOpReadVariableOp?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/v*
_output_shapes
:
*
dtype0
P
ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
e
ReadVariableOpReadVariableOp
Variable_1^Variable_1/Assign*
_output_shapes
: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallReadVariableOpStatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_383799
<
NoOpNoOp^StatefulPartitionedCall_1^Variable_1/Assign
?O
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?O
value?OB?O B?O
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
|
	_func
	_callable

Variable:0
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
y
forward_layer
backward_layer
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?
-iter

.beta_1

/beta_2
	0decay
1learning_ratem?m?m?'m?(m?2m?3m?4m?5m?6m?7m?v?v?v?'v?(v?2v?3v?4v?5v?6v?7v?
N
0
21
32
43
54
65
76
7
8
'9
(10
N
0
21
32
43
54
65
76
7
8
'9
(10
 
?
	variables
8layer_regularization_losses
9layer_metrics
	trainable_variables

regularization_losses

:layers
;metrics
<non_trainable_variables
 
?
=_vocabulary_file

>_table

embeddings
?	variables
?trainable_variables
@
signatures
#A_self_saveable_object_factories
XV
VARIABLE_VALUEVariable:layer_with_weights-0/Variable:0/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
Blayer_regularization_losses
Clayer_metrics
trainable_variables
regularization_losses

Dlayers
Emetrics
Fnon_trainable_variables
 
 
 
?
	variables
Glayer_regularization_losses
Hlayer_metrics
trainable_variables
regularization_losses

Ilayers
Jmetrics
Knon_trainable_variables
l
Lcell
M
state_spec
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
l
Rcell
S
state_spec
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
*
20
31
42
53
64
75
*
20
31
42
53
64
75
 
?
	variables
Xlayer_regularization_losses
Ylayer_metrics
trainable_variables
regularization_losses

Zlayers
[metrics
\non_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
]layer_regularization_losses
^layer_metrics
 trainable_variables
!regularization_losses

_layers
`metrics
anon_trainable_variables
 
 
 
?
#	variables
blayer_regularization_losses
clayer_metrics
$trainable_variables
%regularization_losses

dlayers
emetrics
fnon_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
)	variables
glayer_regularization_losses
hlayer_metrics
*trainable_variables
+regularization_losses

ilayers
jmetrics
knon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE9bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUECbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE:bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEDbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
 
*
0
1
2
3
4
5

l0
m1
n2
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
~

2kernel
3recurrent_kernel
4bias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
 

20
31
42

20
31
42
 
?
N	variables
slayer_regularization_losses
tlayer_metrics
Otrainable_variables

ustates
Pregularization_losses

vlayers
wmetrics
xnon_trainable_variables
~

5kernel
6recurrent_kernel
7bias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
 

50
61
72

50
61
72
 
?
T	variables
}layer_regularization_losses
~layer_metrics
Utrainable_variables

states
Vregularization_losses
?layers
?metrics
?non_trainable_variables
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api

20
31
42

20
31
42
 
?
o	variables
 ?layer_regularization_losses
?layer_metrics
ptrainable_variables
qregularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 

L0
 
 

50
61
72

50
61
72
 
?
y	variables
 ?layer_regularization_losses
?layer_metrics
ztrainable_variables
{regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 

R0
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
{y
VARIABLE_VALUEAdam/Variable/mVlayer_with_weights-0/Variable:0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEJAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEKAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Variable/vVlayer_with_weights-0/Variable:0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEJAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEKAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
p
serving_default_InputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_2StatefulPartitionedCallserving_default_InputStatefulPartitionedCallConstVariable9bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel7bidirectional/forward_simple_rnn/simple_rnn_cell_1/biasCbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel:bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel8bidirectional/backward_simple_rnn/simple_rnn_cell_2/biasDbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kerneldense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_380989
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpMbidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/Read/ReadVariableOpWbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/Read/ReadVariableOpKbidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/Read/ReadVariableOpNbidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/Read/ReadVariableOpXbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOpLbidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp#Adam/Variable/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOpTAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/m/Read/ReadVariableOp^Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/m/Read/ReadVariableOpRAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/m/Read/ReadVariableOpUAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/m/Read/ReadVariableOp_Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/m/Read/ReadVariableOpSAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/m/Read/ReadVariableOp#Adam/Variable/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpTAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/v/Read/ReadVariableOp^Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/v/Read/ReadVariableOpRAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/v/Read/ReadVariableOpUAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/v/Read/ReadVariableOp_Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/v/Read/ReadVariableOpSAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/v/Read/ReadVariableOpConst_1*9
Tin2
02.	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_383966
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filenameVariabledense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate9bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernelCbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel7bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias:bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernelDbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel8bidirectional/backward_simple_rnn/simple_rnn_cell_2/biastotalcounttotal_1count_1total_2count_2Adam/Variable/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/m@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/mJAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/m>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/mAAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/mKAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/m?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/mAdam/Variable/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/vJAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/v>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/vAAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/vKAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/v?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/v*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_384108??7
?
?
4__inference_backward_simple_rnn_layer_call_fn_383187
inputs_0
unknown:

	unknown_0:

	unknown_1:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_3791732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?1
?
while_body_383257
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?

?
2__inference_simple_rnn_cell_2_layer_call_fn_383741

inputs
states_0
unknown:

	unknown_0:

	unknown_1:


identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_3789212
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0
?
d
__inference_<lambda>_383799
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_2222792
StatefulPartitionedCallS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
?1
?
while_body_382875
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_1/MatMul/ReadVariableOp?/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOp?
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_1/MatMul?
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_1/BiasAdd?
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_1/MatMul_1?
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/add?
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_380694

inputsU
Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
R
Dforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
W
Eforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

V
Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
S
Ebackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
X
Fbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?backward_simple_rnn/while?;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?forward_simple_rnn/whilej
forward_simple_rnn/ShapeShapeinputs*
T0*
_output_shapes
:2
forward_simple_rnn/Shape?
&forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_simple_rnn/strided_slice/stack?
(forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_1?
(forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_2?
 forward_simple_rnn/strided_sliceStridedSlice!forward_simple_rnn/Shape:output:0/forward_simple_rnn/strided_slice/stack:output:01forward_simple_rnn/strided_slice/stack_1:output:01forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_simple_rnn/strided_slice?
forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2 
forward_simple_rnn/zeros/mul/y?
forward_simple_rnn/zeros/mulMul)forward_simple_rnn/strided_slice:output:0'forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/mul?
forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
forward_simple_rnn/zeros/Less/y?
forward_simple_rnn/zeros/LessLess forward_simple_rnn/zeros/mul:z:0(forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/Less?
!forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!forward_simple_rnn/zeros/packed/1?
forward_simple_rnn/zeros/packedPack)forward_simple_rnn/strided_slice:output:0*forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_simple_rnn/zeros/packed?
forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
forward_simple_rnn/zeros/Const?
forward_simple_rnn/zerosFill(forward_simple_rnn/zeros/packed:output:0'forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
forward_simple_rnn/zeros?
!forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_simple_rnn/transpose/perm?
forward_simple_rnn/transpose	Transposeinputs*forward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
forward_simple_rnn/transpose?
forward_simple_rnn/Shape_1Shape forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
forward_simple_rnn/Shape_1?
(forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_1/stack?
*forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_1?
*forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_2?
"forward_simple_rnn/strided_slice_1StridedSlice#forward_simple_rnn/Shape_1:output:01forward_simple_rnn/strided_slice_1/stack:output:03forward_simple_rnn/strided_slice_1/stack_1:output:03forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_1?
.forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.forward_simple_rnn/TensorArrayV2/element_shape?
 forward_simple_rnn/TensorArrayV2TensorListReserve7forward_simple_rnn/TensorArrayV2/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_simple_rnn/TensorArrayV2?
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2J
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor forward_simple_rnn/transpose:y:0Qforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
(forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_2/stack?
*forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_1?
*forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_2?
"forward_simple_rnn/strided_slice_2StridedSlice forward_simple_rnn/transpose:y:01forward_simple_rnn/strided_slice_2/stack:output:03forward_simple_rnn/strided_slice_2/stack_1:output:03forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_2?
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpCforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02<
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
+forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul+forward_simple_rnn/strided_slice_2:output:0Bforward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2-
+forward_simple_rnn/simple_rnn_cell_1/MatMul?
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02=
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
,forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAdd5forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0Cforward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02>
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMul!forward_simple_rnn/zeros:output:0Dforward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
(forward_simple_rnn/simple_rnn_cell_1/addAddV25forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:07forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2*
(forward_simple_rnn/simple_rnn_cell_1/add?
)forward_simple_rnn/simple_rnn_cell_1/TanhTanh,forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2+
)forward_simple_rnn/simple_rnn_cell_1/Tanh?
0forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0forward_simple_rnn/TensorArrayV2_1/element_shape?
"forward_simple_rnn/TensorArrayV2_1TensorListReserve9forward_simple_rnn/TensorArrayV2_1/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"forward_simple_rnn/TensorArrayV2_1t
forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_simple_rnn/time?
+forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+forward_simple_rnn/while/maximum_iterations?
%forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%forward_simple_rnn/while/loop_counter?
forward_simple_rnn/whileWhile.forward_simple_rnn/while/loop_counter:output:04forward_simple_rnn/while/maximum_iterations:output:0 forward_simple_rnn/time:output:0+forward_simple_rnn/TensorArrayV2_1:handle:0!forward_simple_rnn/zeros:output:0+forward_simple_rnn/strided_slice_1:output:0Jforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourceDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourceEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$forward_simple_rnn_while_body_380516*0
cond(R&
$forward_simple_rnn_while_cond_380515*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
forward_simple_rnn/while?
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2E
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
5forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack!forward_simple_rnn/while:output:3Lforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype027
5forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
(forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(forward_simple_rnn/strided_slice_3/stack?
*forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*forward_simple_rnn/strided_slice_3/stack_1?
*forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_3/stack_2?
"forward_simple_rnn/strided_slice_3StridedSlice>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:01forward_simple_rnn/strided_slice_3/stack:output:03forward_simple_rnn/strided_slice_3/stack_1:output:03forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_3?
#forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#forward_simple_rnn/transpose_1/perm?
forward_simple_rnn/transpose_1	Transpose>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0,forward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2 
forward_simple_rnn/transpose_1l
backward_simple_rnn/ShapeShapeinputs*
T0*
_output_shapes
:2
backward_simple_rnn/Shape?
'backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_simple_rnn/strided_slice/stack?
)backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_1?
)backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_2?
!backward_simple_rnn/strided_sliceStridedSlice"backward_simple_rnn/Shape:output:00backward_simple_rnn/strided_slice/stack:output:02backward_simple_rnn/strided_slice/stack_1:output:02backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_simple_rnn/strided_slice?
backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2!
backward_simple_rnn/zeros/mul/y?
backward_simple_rnn/zeros/mulMul*backward_simple_rnn/strided_slice:output:0(backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/zeros/mul?
 backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 backward_simple_rnn/zeros/Less/y?
backward_simple_rnn/zeros/LessLess!backward_simple_rnn/zeros/mul:z:0)backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_simple_rnn/zeros/Less?
"backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2$
"backward_simple_rnn/zeros/packed/1?
 backward_simple_rnn/zeros/packedPack*backward_simple_rnn/strided_slice:output:0+backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_simple_rnn/zeros/packed?
backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
backward_simple_rnn/zeros/Const?
backward_simple_rnn/zerosFill)backward_simple_rnn/zeros/packed:output:0(backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
backward_simple_rnn/zeros?
"backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_simple_rnn/transpose/perm?
backward_simple_rnn/transpose	Transposeinputs+backward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
backward_simple_rnn/transpose?
backward_simple_rnn/Shape_1Shape!backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
backward_simple_rnn/Shape_1?
)backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_1/stack?
+backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_1?
+backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_2?
#backward_simple_rnn/strided_slice_1StridedSlice$backward_simple_rnn/Shape_1:output:02backward_simple_rnn/strided_slice_1/stack:output:04backward_simple_rnn/strided_slice_1/stack_1:output:04backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_1?
/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/backward_simple_rnn/TensorArrayV2/element_shape?
!backward_simple_rnn/TensorArrayV2TensorListReserve8backward_simple_rnn/TensorArrayV2/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_simple_rnn/TensorArrayV2?
"backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2$
"backward_simple_rnn/ReverseV2/axis?
backward_simple_rnn/ReverseV2	ReverseV2!backward_simple_rnn/transpose:y:0+backward_simple_rnn/ReverseV2/axis:output:0*
T0*+
_output_shapes
:?????????2
backward_simple_rnn/ReverseV2?
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2K
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor&backward_simple_rnn/ReverseV2:output:0Rbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
)backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_2/stack?
+backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_1?
+backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_2?
#backward_simple_rnn/strided_slice_2StridedSlice!backward_simple_rnn/transpose:y:02backward_simple_rnn/strided_slice_2/stack:output:04backward_simple_rnn/strided_slice_2/stack_1:output:04backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_2?
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpDbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02=
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
,backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul,backward_simple_rnn/strided_slice_2:output:0Cbackward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,backward_simple_rnn/simple_rnn_cell_2/MatMul?
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02>
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
-backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd6backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0Dbackward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMul"backward_simple_rnn/zeros:output:0Ebackward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
20
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
)backward_simple_rnn/simple_rnn_cell_2/addAddV26backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:08backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2+
)backward_simple_rnn/simple_rnn_cell_2/add?
*backward_simple_rnn/simple_rnn_cell_2/TanhTanh-backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2,
*backward_simple_rnn/simple_rnn_cell_2/Tanh?
1backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   23
1backward_simple_rnn/TensorArrayV2_1/element_shape?
#backward_simple_rnn/TensorArrayV2_1TensorListReserve:backward_simple_rnn/TensorArrayV2_1/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#backward_simple_rnn/TensorArrayV2_1v
backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_simple_rnn/time?
,backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,backward_simple_rnn/while/maximum_iterations?
&backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_simple_rnn/while/loop_counter?
backward_simple_rnn/whileWhile/backward_simple_rnn/while/loop_counter:output:05backward_simple_rnn/while/maximum_iterations:output:0!backward_simple_rnn/time:output:0,backward_simple_rnn/TensorArrayV2_1:handle:0"backward_simple_rnn/zeros:output:0,backward_simple_rnn/strided_slice_1:output:0Kbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%backward_simple_rnn_while_body_380626*1
cond)R'
%backward_simple_rnn_while_cond_380625*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
backward_simple_rnn/while?
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2F
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
6backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack"backward_simple_rnn/while:output:3Mbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype028
6backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
)backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)backward_simple_rnn/strided_slice_3/stack?
+backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+backward_simple_rnn/strided_slice_3/stack_1?
+backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_3/stack_2?
#backward_simple_rnn/strided_slice_3StridedSlice?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:02backward_simple_rnn/strided_slice_3/stack:output:04backward_simple_rnn/strided_slice_3/stack_1:output:04backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_3?
$backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$backward_simple_rnn/transpose_1/perm?
backward_simple_rnn/transpose_1	Transpose?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0-backward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2!
backward_simple_rnn/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2+forward_simple_rnn/strided_slice_3:output:0,backward_simple_rnn/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp=^backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<^backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp>^backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp^backward_simple_rnn/while<^forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp=^forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp^forward_simple_rnn/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2|
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2z
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2~
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp26
backward_simple_rnn/whilebackward_simple_rnn/while2z
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp24
forward_simple_rnn/whileforward_simple_rnn/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_378571
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_378571___redundant_placeholder04
0while_while_cond_378571___redundant_placeholder14
0while_while_cond_378571___redundant_placeholder24
0while_while_cond_378571___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?G
?
%backward_simple_rnn_while_body_381862D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2C
?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0
{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
[
Mbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
`
Nbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

&
"backward_simple_rnn_while_identity(
$backward_simple_rnn_while_identity_1(
$backward_simple_rnn_while_identity_2(
$backward_simple_rnn_while_identity_3(
$backward_simple_rnn_while_identity_4A
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1}
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor\
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
Y
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????2M
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0%backward_simple_rnn_while_placeholderTbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02C
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
2backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulDbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Ibackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02D
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd<backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Jbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02E
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul'backward_simple_rnn_while_placeholder_2Kbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
26
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2<backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0>backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
21
/backward_simple_rnn/while/simple_rnn_cell_2/add?
0backward_simple_rnn/while/simple_rnn_cell_2/TanhTanh3backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
22
0backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'backward_simple_rnn_while_placeholder_1%backward_simple_rnn_while_placeholder4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02@
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_simple_rnn/while/add/y?
backward_simple_rnn/while/addAddV2%backward_simple_rnn_while_placeholder(backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/while/add?
!backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!backward_simple_rnn/while/add_1/y?
backward_simple_rnn/while/add_1AddV2@backward_simple_rnn_while_backward_simple_rnn_while_loop_counter*backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
backward_simple_rnn/while/add_1?
"backward_simple_rnn/while/IdentityIdentity#backward_simple_rnn/while/add_1:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity?
$backward_simple_rnn/while/Identity_1IdentityFbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_1?
$backward_simple_rnn/while/Identity_2Identity!backward_simple_rnn/while/add:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_2?
$backward_simple_rnn/while/Identity_3IdentityNbackward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_3?
$backward_simple_rnn/while/Identity_4Identity4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0^backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2&
$backward_simple_rnn/while/Identity_4?
backward_simple_rnn/while/NoOpNoOpC^backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpB^backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpD^backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
backward_simple_rnn/while/NoOp"?
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0"U
$backward_simple_rnn_while_identity_1-backward_simple_rnn/while/Identity_1:output:0"U
$backward_simple_rnn_while_identity_2-backward_simple_rnn/while/Identity_2:output:0"U
$backward_simple_rnn_while_identity_3-backward_simple_rnn/while/Identity_3:output:0"U
$backward_simple_rnn_while_identity_4-backward_simple_rnn/while/Identity_4:output:0"?
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpBbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpAbackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpCbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?>
?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_378635

inputs*
simple_rnn_cell_1_378560:
&
simple_rnn_cell_1_378562:
*
simple_rnn_cell_1_378564:


identity??)simple_rnn_cell_1/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
)simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_1_378560simple_rnn_cell_1_378562simple_rnn_cell_1_378564*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_3785052+
)simple_rnn_cell_1/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_1_378560simple_rnn_cell_1_378562simple_rnn_cell_1_378564*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_378572*
condR
while_cond_378571*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp*^simple_rnn_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2V
)simple_rnn_cell_1/StatefulPartitionedCall)simple_rnn_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?1
?
while_body_379461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_1/MatMul/ReadVariableOp?/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOp?
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_1/MatMul?
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_1/BiasAdd?
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_1/MatMul_1?
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/add?
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
3bidirectional_backward_simple_rnn_while_cond_381481`
\bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_loop_counterf
bbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_maximum_iterations7
3bidirectional_backward_simple_rnn_while_placeholder9
5bidirectional_backward_simple_rnn_while_placeholder_19
5bidirectional_backward_simple_rnn_while_placeholder_2b
^bidirectional_backward_simple_rnn_while_less_bidirectional_backward_simple_rnn_strided_slice_1x
tbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_cond_381481___redundant_placeholder0x
tbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_cond_381481___redundant_placeholder1x
tbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_cond_381481___redundant_placeholder2x
tbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_cond_381481___redundant_placeholder34
0bidirectional_backward_simple_rnn_while_identity
?
,bidirectional/backward_simple_rnn/while/LessLess3bidirectional_backward_simple_rnn_while_placeholder^bidirectional_backward_simple_rnn_while_less_bidirectional_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2.
,bidirectional/backward_simple_rnn/while/Less?
0bidirectional/backward_simple_rnn/while/IdentityIdentity0bidirectional/backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 22
0bidirectional/backward_simple_rnn/while/Identity"m
0bidirectional_backward_simple_rnn_while_identity9bidirectional/backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?1
?
while_body_382763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_1/MatMul/ReadVariableOp?/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOp?
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_1/MatMul?
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_1/BiasAdd?
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_1/MatMul_1?
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/add?
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_383098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_383098___redundant_placeholder04
0while_while_cond_383098___redundant_placeholder14
0while_while_cond_383098___redundant_placeholder24
0while_while_cond_383098___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
$forward_simple_rnn_while_cond_381975B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2D
@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_381975___redundant_placeholder0Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_381975___redundant_placeholder1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_381975___redundant_placeholder2Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_381975___redundant_placeholder3%
!forward_simple_rnn_while_identity
?
forward_simple_rnn/while/LessLess$forward_simple_rnn_while_placeholder@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
forward_simple_rnn/while/Less?
!forward_simple_rnn/while/IdentityIdentity!forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
A__inference_embed_layer_call_and_return_conditional_losses_380740

inputs
unknown
	unknown_0	
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_2217052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
??
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_382378

inputsU
Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
R
Dforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
W
Eforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

V
Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
S
Ebackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
X
Fbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?backward_simple_rnn/while?;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?forward_simple_rnn/whilej
forward_simple_rnn/ShapeShapeinputs*
T0*
_output_shapes
:2
forward_simple_rnn/Shape?
&forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_simple_rnn/strided_slice/stack?
(forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_1?
(forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_2?
 forward_simple_rnn/strided_sliceStridedSlice!forward_simple_rnn/Shape:output:0/forward_simple_rnn/strided_slice/stack:output:01forward_simple_rnn/strided_slice/stack_1:output:01forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_simple_rnn/strided_slice?
forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2 
forward_simple_rnn/zeros/mul/y?
forward_simple_rnn/zeros/mulMul)forward_simple_rnn/strided_slice:output:0'forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/mul?
forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
forward_simple_rnn/zeros/Less/y?
forward_simple_rnn/zeros/LessLess forward_simple_rnn/zeros/mul:z:0(forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/Less?
!forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!forward_simple_rnn/zeros/packed/1?
forward_simple_rnn/zeros/packedPack)forward_simple_rnn/strided_slice:output:0*forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_simple_rnn/zeros/packed?
forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
forward_simple_rnn/zeros/Const?
forward_simple_rnn/zerosFill(forward_simple_rnn/zeros/packed:output:0'forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
forward_simple_rnn/zeros?
!forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_simple_rnn/transpose/perm?
forward_simple_rnn/transpose	Transposeinputs*forward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
forward_simple_rnn/transpose?
forward_simple_rnn/Shape_1Shape forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
forward_simple_rnn/Shape_1?
(forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_1/stack?
*forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_1?
*forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_2?
"forward_simple_rnn/strided_slice_1StridedSlice#forward_simple_rnn/Shape_1:output:01forward_simple_rnn/strided_slice_1/stack:output:03forward_simple_rnn/strided_slice_1/stack_1:output:03forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_1?
.forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.forward_simple_rnn/TensorArrayV2/element_shape?
 forward_simple_rnn/TensorArrayV2TensorListReserve7forward_simple_rnn/TensorArrayV2/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_simple_rnn/TensorArrayV2?
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2J
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor forward_simple_rnn/transpose:y:0Qforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
(forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_2/stack?
*forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_1?
*forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_2?
"forward_simple_rnn/strided_slice_2StridedSlice forward_simple_rnn/transpose:y:01forward_simple_rnn/strided_slice_2/stack:output:03forward_simple_rnn/strided_slice_2/stack_1:output:03forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_2?
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpCforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02<
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
+forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul+forward_simple_rnn/strided_slice_2:output:0Bforward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2-
+forward_simple_rnn/simple_rnn_cell_1/MatMul?
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02=
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
,forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAdd5forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0Cforward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02>
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMul!forward_simple_rnn/zeros:output:0Dforward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
(forward_simple_rnn/simple_rnn_cell_1/addAddV25forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:07forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2*
(forward_simple_rnn/simple_rnn_cell_1/add?
)forward_simple_rnn/simple_rnn_cell_1/TanhTanh,forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2+
)forward_simple_rnn/simple_rnn_cell_1/Tanh?
0forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0forward_simple_rnn/TensorArrayV2_1/element_shape?
"forward_simple_rnn/TensorArrayV2_1TensorListReserve9forward_simple_rnn/TensorArrayV2_1/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"forward_simple_rnn/TensorArrayV2_1t
forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_simple_rnn/time?
+forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+forward_simple_rnn/while/maximum_iterations?
%forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%forward_simple_rnn/while/loop_counter?
forward_simple_rnn/whileWhile.forward_simple_rnn/while/loop_counter:output:04forward_simple_rnn/while/maximum_iterations:output:0 forward_simple_rnn/time:output:0+forward_simple_rnn/TensorArrayV2_1:handle:0!forward_simple_rnn/zeros:output:0+forward_simple_rnn/strided_slice_1:output:0Jforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourceDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourceEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$forward_simple_rnn_while_body_382200*0
cond(R&
$forward_simple_rnn_while_cond_382199*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
forward_simple_rnn/while?
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2E
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
5forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack!forward_simple_rnn/while:output:3Lforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype027
5forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
(forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(forward_simple_rnn/strided_slice_3/stack?
*forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*forward_simple_rnn/strided_slice_3/stack_1?
*forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_3/stack_2?
"forward_simple_rnn/strided_slice_3StridedSlice>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:01forward_simple_rnn/strided_slice_3/stack:output:03forward_simple_rnn/strided_slice_3/stack_1:output:03forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_3?
#forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#forward_simple_rnn/transpose_1/perm?
forward_simple_rnn/transpose_1	Transpose>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0,forward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2 
forward_simple_rnn/transpose_1l
backward_simple_rnn/ShapeShapeinputs*
T0*
_output_shapes
:2
backward_simple_rnn/Shape?
'backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_simple_rnn/strided_slice/stack?
)backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_1?
)backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_2?
!backward_simple_rnn/strided_sliceStridedSlice"backward_simple_rnn/Shape:output:00backward_simple_rnn/strided_slice/stack:output:02backward_simple_rnn/strided_slice/stack_1:output:02backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_simple_rnn/strided_slice?
backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2!
backward_simple_rnn/zeros/mul/y?
backward_simple_rnn/zeros/mulMul*backward_simple_rnn/strided_slice:output:0(backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/zeros/mul?
 backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 backward_simple_rnn/zeros/Less/y?
backward_simple_rnn/zeros/LessLess!backward_simple_rnn/zeros/mul:z:0)backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_simple_rnn/zeros/Less?
"backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2$
"backward_simple_rnn/zeros/packed/1?
 backward_simple_rnn/zeros/packedPack*backward_simple_rnn/strided_slice:output:0+backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_simple_rnn/zeros/packed?
backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
backward_simple_rnn/zeros/Const?
backward_simple_rnn/zerosFill)backward_simple_rnn/zeros/packed:output:0(backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
backward_simple_rnn/zeros?
"backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_simple_rnn/transpose/perm?
backward_simple_rnn/transpose	Transposeinputs+backward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
backward_simple_rnn/transpose?
backward_simple_rnn/Shape_1Shape!backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
backward_simple_rnn/Shape_1?
)backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_1/stack?
+backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_1?
+backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_2?
#backward_simple_rnn/strided_slice_1StridedSlice$backward_simple_rnn/Shape_1:output:02backward_simple_rnn/strided_slice_1/stack:output:04backward_simple_rnn/strided_slice_1/stack_1:output:04backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_1?
/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/backward_simple_rnn/TensorArrayV2/element_shape?
!backward_simple_rnn/TensorArrayV2TensorListReserve8backward_simple_rnn/TensorArrayV2/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_simple_rnn/TensorArrayV2?
"backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2$
"backward_simple_rnn/ReverseV2/axis?
backward_simple_rnn/ReverseV2	ReverseV2!backward_simple_rnn/transpose:y:0+backward_simple_rnn/ReverseV2/axis:output:0*
T0*+
_output_shapes
:?????????2
backward_simple_rnn/ReverseV2?
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2K
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor&backward_simple_rnn/ReverseV2:output:0Rbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
)backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_2/stack?
+backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_1?
+backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_2?
#backward_simple_rnn/strided_slice_2StridedSlice!backward_simple_rnn/transpose:y:02backward_simple_rnn/strided_slice_2/stack:output:04backward_simple_rnn/strided_slice_2/stack_1:output:04backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_2?
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpDbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02=
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
,backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul,backward_simple_rnn/strided_slice_2:output:0Cbackward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,backward_simple_rnn/simple_rnn_cell_2/MatMul?
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02>
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
-backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd6backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0Dbackward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMul"backward_simple_rnn/zeros:output:0Ebackward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
20
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
)backward_simple_rnn/simple_rnn_cell_2/addAddV26backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:08backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2+
)backward_simple_rnn/simple_rnn_cell_2/add?
*backward_simple_rnn/simple_rnn_cell_2/TanhTanh-backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2,
*backward_simple_rnn/simple_rnn_cell_2/Tanh?
1backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   23
1backward_simple_rnn/TensorArrayV2_1/element_shape?
#backward_simple_rnn/TensorArrayV2_1TensorListReserve:backward_simple_rnn/TensorArrayV2_1/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#backward_simple_rnn/TensorArrayV2_1v
backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_simple_rnn/time?
,backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,backward_simple_rnn/while/maximum_iterations?
&backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_simple_rnn/while/loop_counter?
backward_simple_rnn/whileWhile/backward_simple_rnn/while/loop_counter:output:05backward_simple_rnn/while/maximum_iterations:output:0!backward_simple_rnn/time:output:0,backward_simple_rnn/TensorArrayV2_1:handle:0"backward_simple_rnn/zeros:output:0,backward_simple_rnn/strided_slice_1:output:0Kbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%backward_simple_rnn_while_body_382310*1
cond)R'
%backward_simple_rnn_while_cond_382309*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
backward_simple_rnn/while?
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2F
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
6backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack"backward_simple_rnn/while:output:3Mbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype028
6backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
)backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)backward_simple_rnn/strided_slice_3/stack?
+backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+backward_simple_rnn/strided_slice_3/stack_1?
+backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_3/stack_2?
#backward_simple_rnn/strided_slice_3StridedSlice?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:02backward_simple_rnn/strided_slice_3/stack:output:04backward_simple_rnn/strided_slice_3/stack_1:output:04backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_3?
$backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$backward_simple_rnn/transpose_1/perm?
backward_simple_rnn/transpose_1	Transpose?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0-backward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2!
backward_simple_rnn/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2+forward_simple_rnn/strided_slice_3:output:0,backward_simple_rnn/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp=^backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<^backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp>^backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp^backward_simple_rnn/while<^forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp=^forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp^forward_simple_rnn/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2|
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2z
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2~
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp26
backward_simple_rnn/whilebackward_simple_rnn/while2z
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp24
forward_simple_rnn/whileforward_simple_rnn/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_383710

inputs
states_00
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????
2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityg

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0
?
?
!__inference_signature_wrapper_123
	sentences"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2	2
statefulpartitionedcall_args_3:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	sentencesstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2	*
Tout
2**
_gradient_op_typePartitionedCall-119*
_output_shapes
:*!
fR
__inference___call___1152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
::::22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
::

_output_shapes
:
?
?
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_381051

inputs
unknown
	unknown_0	
	unknown_1:
??
	unknown_2:

	unknown_3:

	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:
	unknown_9:

unknown_10:

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_3808182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_380094

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_379801

inputsB
0simple_rnn_cell_2_matmul_readvariableop_resource:
?
1simple_rnn_cell_2_biasadd_readvariableop_resource:
D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	ReverseV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_379735*
condR
while_cond_379734*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?F
?
$forward_simple_rnn_while_body_382424B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2A
=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0}
yforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
Z
Lforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
_
Mforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

%
!forward_simple_rnn_while_identity'
#forward_simple_rnn_while_identity_1'
#forward_simple_rnn_while_identity_2'
#forward_simple_rnn_while_identity_3'
#forward_simple_rnn_while_identity_4?
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1{
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor[
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
X
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0$forward_simple_rnn_while_placeholderSforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02>
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02B
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
1forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulCforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Hforward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
23
1forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02C
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAdd;forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0Iforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02D
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMul&forward_simple_rnn_while_placeholder_2Jforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
.forward_simple_rnn/while/simple_rnn_cell_1/addAddV2;forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0=forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
20
.forward_simple_rnn/while/simple_rnn_cell_1/add?
/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanh2forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
21
/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&forward_simple_rnn_while_placeholder_1$forward_simple_rnn_while_placeholder3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_simple_rnn/while/add/y?
forward_simple_rnn/while/addAddV2$forward_simple_rnn_while_placeholder'forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/while/add?
 forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 forward_simple_rnn/while/add_1/y?
forward_simple_rnn/while/add_1AddV2>forward_simple_rnn_while_forward_simple_rnn_while_loop_counter)forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
forward_simple_rnn/while/add_1?
!forward_simple_rnn/while/IdentityIdentity"forward_simple_rnn/while/add_1:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity?
#forward_simple_rnn/while/Identity_1IdentityDforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_1?
#forward_simple_rnn/while/Identity_2Identity forward_simple_rnn/while/add:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_2?
#forward_simple_rnn/while/Identity_3IdentityMforward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_3?
#forward_simple_rnn/while/Identity_4Identity3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0^forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2%
#forward_simple_rnn/while/Identity_4?
forward_simple_rnn/while/NoOpNoOpB^forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpA^forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpC^forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_simple_rnn/while/NoOp"|
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0"S
#forward_simple_rnn_while_identity_1,forward_simple_rnn/while/Identity_1:output:0"S
#forward_simple_rnn_while_identity_2,forward_simple_rnn/while/Identity_2:output:0"S
#forward_simple_rnn_while_identity_3,forward_simple_rnn/while/Identity_3:output:0"S
#forward_simple_rnn_while_identity_4,forward_simple_rnn/while/Identity_4:output:0"?
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourceLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensoryforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpAforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpBforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?I
?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_379527

inputsB
0simple_rnn_cell_1_matmul_readvariableop_resource:
?
1simple_rnn_cell_1_biasadd_readvariableop_resource:
D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_1/BiasAdd/ReadVariableOp?'simple_rnn_cell_1/MatMul/ReadVariableOp?)simple_rnn_cell_1/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOp?
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul?
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOp?
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/BiasAdd?
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOp?
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul_1?
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/add?
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_379461*
condR
while_cond_379460*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
2__inference_simple_rnn_cell_1_layer_call_fn_383679

inputs
states_0
unknown:

	unknown_0:

	unknown_1:


identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_3783852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0
?
?
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_383727

inputs
states_00
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????
2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityg

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0
?G
?
%backward_simple_rnn_while_body_382310D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2C
?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0
{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
[
Mbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
`
Nbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

&
"backward_simple_rnn_while_identity(
$backward_simple_rnn_while_identity_1(
$backward_simple_rnn_while_identity_2(
$backward_simple_rnn_while_identity_3(
$backward_simple_rnn_while_identity_4A
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1}
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor\
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
Y
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0%backward_simple_rnn_while_placeholderTbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02C
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
2backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulDbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Ibackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02D
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd<backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Jbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02E
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul'backward_simple_rnn_while_placeholder_2Kbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
26
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2<backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0>backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
21
/backward_simple_rnn/while/simple_rnn_cell_2/add?
0backward_simple_rnn/while/simple_rnn_cell_2/TanhTanh3backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
22
0backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'backward_simple_rnn_while_placeholder_1%backward_simple_rnn_while_placeholder4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02@
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_simple_rnn/while/add/y?
backward_simple_rnn/while/addAddV2%backward_simple_rnn_while_placeholder(backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/while/add?
!backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!backward_simple_rnn/while/add_1/y?
backward_simple_rnn/while/add_1AddV2@backward_simple_rnn_while_backward_simple_rnn_while_loop_counter*backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
backward_simple_rnn/while/add_1?
"backward_simple_rnn/while/IdentityIdentity#backward_simple_rnn/while/add_1:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity?
$backward_simple_rnn/while/Identity_1IdentityFbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_1?
$backward_simple_rnn/while/Identity_2Identity!backward_simple_rnn/while/add:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_2?
$backward_simple_rnn/while/Identity_3IdentityNbackward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_3?
$backward_simple_rnn/while/Identity_4Identity4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0^backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2&
$backward_simple_rnn/while/Identity_4?
backward_simple_rnn/while/NoOpNoOpC^backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpB^backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpD^backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
backward_simple_rnn/while/NoOp"?
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0"U
$backward_simple_rnn_while_identity_1-backward_simple_rnn/while/Identity_1:output:0"U
$backward_simple_rnn_while_identity_2-backward_simple_rnn/while/Identity_2:output:0"U
$backward_simple_rnn_while_identity_3-backward_simple_rnn/while/Identity_3:output:0"U
$backward_simple_rnn_while_identity_4-backward_simple_rnn/while/Identity_4:output:0"?
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpBbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpAbackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpCbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?G
?
%backward_simple_rnn_while_body_380626D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2C
?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0
{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
[
Mbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
`
Nbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

&
"backward_simple_rnn_while_identity(
$backward_simple_rnn_while_identity_1(
$backward_simple_rnn_while_identity_2(
$backward_simple_rnn_while_identity_3(
$backward_simple_rnn_while_identity_4A
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1}
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor\
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
Y
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0%backward_simple_rnn_while_placeholderTbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02C
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
2backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulDbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Ibackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02D
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd<backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Jbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02E
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul'backward_simple_rnn_while_placeholder_2Kbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
26
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2<backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0>backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
21
/backward_simple_rnn/while/simple_rnn_cell_2/add?
0backward_simple_rnn/while/simple_rnn_cell_2/TanhTanh3backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
22
0backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'backward_simple_rnn_while_placeholder_1%backward_simple_rnn_while_placeholder4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02@
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_simple_rnn/while/add/y?
backward_simple_rnn/while/addAddV2%backward_simple_rnn_while_placeholder(backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/while/add?
!backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!backward_simple_rnn/while/add_1/y?
backward_simple_rnn/while/add_1AddV2@backward_simple_rnn_while_backward_simple_rnn_while_loop_counter*backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
backward_simple_rnn/while/add_1?
"backward_simple_rnn/while/IdentityIdentity#backward_simple_rnn/while/add_1:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity?
$backward_simple_rnn/while/Identity_1IdentityFbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_1?
$backward_simple_rnn/while/Identity_2Identity!backward_simple_rnn/while/add:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_2?
$backward_simple_rnn/while/Identity_3IdentityNbackward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_3?
$backward_simple_rnn/while/Identity_4Identity4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0^backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2&
$backward_simple_rnn/while/Identity_4?
backward_simple_rnn/while/NoOpNoOpC^backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpB^backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpD^backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
backward_simple_rnn/while/NoOp"?
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0"U
$backward_simple_rnn_while_identity_1-backward_simple_rnn/while/Identity_1:output:0"U
$backward_simple_rnn_while_identity_2-backward_simple_rnn/while/Identity_2:output:0"U
$backward_simple_rnn_while_identity_3-backward_simple_rnn/while/Identity_3:output:0"U
$backward_simple_rnn_while_identity_4-backward_simple_rnn/while/Identity_4:output:0"?
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpBbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpAbackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpCbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_379109
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_379109___redundant_placeholder04
0while_while_cond_379109___redundant_placeholder14
0while_while_cond_379109___redundant_placeholder24
0while_while_cond_379109___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_380878	
input
unknown
	unknown_0	
	unknown_1:
??
	unknown_2:

	unknown_3:

	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:
	unknown_9:

unknown_10:

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_3808182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_nameInput:

_output_shapes
: 
?
?
while_cond_382986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_382986___redundant_placeholder04
0while_while_cond_382986___redundant_placeholder14
0while_while_cond_382986___redundant_placeholder24
0while_while_cond_382986___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
r
)__inference_restored_function_body_222279
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference__initializer_1342
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
Ȥ
?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_381576

inputs
embed_381313
embed_381315	 
embed_381317:
??c
Qbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
`
Rbidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
e
Sbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

d
Rbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
a
Sbidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
f
Tbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:

8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??Jbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Ibidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?Kbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?'bidirectional/backward_simple_rnn/while?Ibidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?Hbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?Jbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?&bidirectional/forward_simple_rnn/while?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embed/StatefulPartitionedCall?
embed/StatefulPartitionedCallStatefulPartitionedCallinputsembed_381313embed_381315embed_381317*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_2217052
embed/StatefulPartitionedCallt
reshape/ShapeShape&embed/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape&embed/StatefulPartitionedCall:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape/Reshape?
&bidirectional/forward_simple_rnn/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2(
&bidirectional/forward_simple_rnn/Shape?
4bidirectional/forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4bidirectional/forward_simple_rnn/strided_slice/stack?
6bidirectional/forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6bidirectional/forward_simple_rnn/strided_slice/stack_1?
6bidirectional/forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6bidirectional/forward_simple_rnn/strided_slice/stack_2?
.bidirectional/forward_simple_rnn/strided_sliceStridedSlice/bidirectional/forward_simple_rnn/Shape:output:0=bidirectional/forward_simple_rnn/strided_slice/stack:output:0?bidirectional/forward_simple_rnn/strided_slice/stack_1:output:0?bidirectional/forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.bidirectional/forward_simple_rnn/strided_slice?
,bidirectional/forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2.
,bidirectional/forward_simple_rnn/zeros/mul/y?
*bidirectional/forward_simple_rnn/zeros/mulMul7bidirectional/forward_simple_rnn/strided_slice:output:05bidirectional/forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2,
*bidirectional/forward_simple_rnn/zeros/mul?
-bidirectional/forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2/
-bidirectional/forward_simple_rnn/zeros/Less/y?
+bidirectional/forward_simple_rnn/zeros/LessLess.bidirectional/forward_simple_rnn/zeros/mul:z:06bidirectional/forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2-
+bidirectional/forward_simple_rnn/zeros/Less?
/bidirectional/forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
21
/bidirectional/forward_simple_rnn/zeros/packed/1?
-bidirectional/forward_simple_rnn/zeros/packedPack7bidirectional/forward_simple_rnn/strided_slice:output:08bidirectional/forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-bidirectional/forward_simple_rnn/zeros/packed?
,bidirectional/forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,bidirectional/forward_simple_rnn/zeros/Const?
&bidirectional/forward_simple_rnn/zerosFill6bidirectional/forward_simple_rnn/zeros/packed:output:05bidirectional/forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2(
&bidirectional/forward_simple_rnn/zeros?
/bidirectional/forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          21
/bidirectional/forward_simple_rnn/transpose/perm?
*bidirectional/forward_simple_rnn/transpose	Transposereshape/Reshape:output:08bidirectional/forward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2,
*bidirectional/forward_simple_rnn/transpose?
(bidirectional/forward_simple_rnn/Shape_1Shape.bidirectional/forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2*
(bidirectional/forward_simple_rnn/Shape_1?
6bidirectional/forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6bidirectional/forward_simple_rnn/strided_slice_1/stack?
8bidirectional/forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_1/stack_1?
8bidirectional/forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_1/stack_2?
0bidirectional/forward_simple_rnn/strided_slice_1StridedSlice1bidirectional/forward_simple_rnn/Shape_1:output:0?bidirectional/forward_simple_rnn/strided_slice_1/stack:output:0Abidirectional/forward_simple_rnn/strided_slice_1/stack_1:output:0Abidirectional/forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0bidirectional/forward_simple_rnn/strided_slice_1?
<bidirectional/forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<bidirectional/forward_simple_rnn/TensorArrayV2/element_shape?
.bidirectional/forward_simple_rnn/TensorArrayV2TensorListReserveEbidirectional/forward_simple_rnn/TensorArrayV2/element_shape:output:09bidirectional/forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.bidirectional/forward_simple_rnn/TensorArrayV2?
Vbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2X
Vbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
Hbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor.bidirectional/forward_simple_rnn/transpose:y:0_bidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02J
Hbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
6bidirectional/forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6bidirectional/forward_simple_rnn/strided_slice_2/stack?
8bidirectional/forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_2/stack_1?
8bidirectional/forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_2/stack_2?
0bidirectional/forward_simple_rnn/strided_slice_2StridedSlice.bidirectional/forward_simple_rnn/transpose:y:0?bidirectional/forward_simple_rnn/strided_slice_2/stack:output:0Abidirectional/forward_simple_rnn/strided_slice_2/stack_1:output:0Abidirectional/forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask22
0bidirectional/forward_simple_rnn/strided_slice_2?
Hbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpQbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02J
Hbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
9bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul9bidirectional/forward_simple_rnn/strided_slice_2:output:0Pbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2;
9bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul?
Ibidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpRbidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02K
Ibidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
:bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAddCbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0Qbidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2<
:bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
Jbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpSbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02L
Jbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
;bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMul/bidirectional/forward_simple_rnn/zeros:output:0Rbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2=
;bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
6bidirectional/forward_simple_rnn/simple_rnn_cell_1/addAddV2Cbidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:0Ebidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
28
6bidirectional/forward_simple_rnn/simple_rnn_cell_1/add?
7bidirectional/forward_simple_rnn/simple_rnn_cell_1/TanhTanh:bidirectional/forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
29
7bidirectional/forward_simple_rnn/simple_rnn_cell_1/Tanh?
>bidirectional/forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2@
>bidirectional/forward_simple_rnn/TensorArrayV2_1/element_shape?
0bidirectional/forward_simple_rnn/TensorArrayV2_1TensorListReserveGbidirectional/forward_simple_rnn/TensorArrayV2_1/element_shape:output:09bidirectional/forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0bidirectional/forward_simple_rnn/TensorArrayV2_1?
%bidirectional/forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2'
%bidirectional/forward_simple_rnn/time?
9bidirectional/forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2;
9bidirectional/forward_simple_rnn/while/maximum_iterations?
3bidirectional/forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 25
3bidirectional/forward_simple_rnn/while/loop_counter?	
&bidirectional/forward_simple_rnn/whileWhile<bidirectional/forward_simple_rnn/while/loop_counter:output:0Bbidirectional/forward_simple_rnn/while/maximum_iterations:output:0.bidirectional/forward_simple_rnn/time:output:09bidirectional/forward_simple_rnn/TensorArrayV2_1:handle:0/bidirectional/forward_simple_rnn/zeros:output:09bidirectional/forward_simple_rnn/strided_slice_1:output:0Xbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Qbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourceRbidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourceSbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *>
body6R4
2bidirectional_forward_simple_rnn_while_body_381372*>
cond6R4
2bidirectional_forward_simple_rnn_while_cond_381371*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2(
&bidirectional/forward_simple_rnn/while?
Qbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2S
Qbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
Cbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack/bidirectional/forward_simple_rnn/while:output:3Zbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02E
Cbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
6bidirectional/forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6bidirectional/forward_simple_rnn/strided_slice_3/stack?
8bidirectional/forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8bidirectional/forward_simple_rnn/strided_slice_3/stack_1?
8bidirectional/forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_3/stack_2?
0bidirectional/forward_simple_rnn/strided_slice_3StridedSliceLbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0?bidirectional/forward_simple_rnn/strided_slice_3/stack:output:0Abidirectional/forward_simple_rnn/strided_slice_3/stack_1:output:0Abidirectional/forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask22
0bidirectional/forward_simple_rnn/strided_slice_3?
1bidirectional/forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          23
1bidirectional/forward_simple_rnn/transpose_1/perm?
,bidirectional/forward_simple_rnn/transpose_1	TransposeLbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0:bidirectional/forward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2.
,bidirectional/forward_simple_rnn/transpose_1?
'bidirectional/backward_simple_rnn/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2)
'bidirectional/backward_simple_rnn/Shape?
5bidirectional/backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5bidirectional/backward_simple_rnn/strided_slice/stack?
7bidirectional/backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7bidirectional/backward_simple_rnn/strided_slice/stack_1?
7bidirectional/backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7bidirectional/backward_simple_rnn/strided_slice/stack_2?
/bidirectional/backward_simple_rnn/strided_sliceStridedSlice0bidirectional/backward_simple_rnn/Shape:output:0>bidirectional/backward_simple_rnn/strided_slice/stack:output:0@bidirectional/backward_simple_rnn/strided_slice/stack_1:output:0@bidirectional/backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/bidirectional/backward_simple_rnn/strided_slice?
-bidirectional/backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2/
-bidirectional/backward_simple_rnn/zeros/mul/y?
+bidirectional/backward_simple_rnn/zeros/mulMul8bidirectional/backward_simple_rnn/strided_slice:output:06bidirectional/backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2-
+bidirectional/backward_simple_rnn/zeros/mul?
.bidirectional/backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?20
.bidirectional/backward_simple_rnn/zeros/Less/y?
,bidirectional/backward_simple_rnn/zeros/LessLess/bidirectional/backward_simple_rnn/zeros/mul:z:07bidirectional/backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2.
,bidirectional/backward_simple_rnn/zeros/Less?
0bidirectional/backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
22
0bidirectional/backward_simple_rnn/zeros/packed/1?
.bidirectional/backward_simple_rnn/zeros/packedPack8bidirectional/backward_simple_rnn/strided_slice:output:09bidirectional/backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.bidirectional/backward_simple_rnn/zeros/packed?
-bidirectional/backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-bidirectional/backward_simple_rnn/zeros/Const?
'bidirectional/backward_simple_rnn/zerosFill7bidirectional/backward_simple_rnn/zeros/packed:output:06bidirectional/backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2)
'bidirectional/backward_simple_rnn/zeros?
0bidirectional/backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          22
0bidirectional/backward_simple_rnn/transpose/perm?
+bidirectional/backward_simple_rnn/transpose	Transposereshape/Reshape:output:09bidirectional/backward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2-
+bidirectional/backward_simple_rnn/transpose?
)bidirectional/backward_simple_rnn/Shape_1Shape/bidirectional/backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2+
)bidirectional/backward_simple_rnn/Shape_1?
7bidirectional/backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7bidirectional/backward_simple_rnn/strided_slice_1/stack?
9bidirectional/backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_1/stack_1?
9bidirectional/backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_1/stack_2?
1bidirectional/backward_simple_rnn/strided_slice_1StridedSlice2bidirectional/backward_simple_rnn/Shape_1:output:0@bidirectional/backward_simple_rnn/strided_slice_1/stack:output:0Bbidirectional/backward_simple_rnn/strided_slice_1/stack_1:output:0Bbidirectional/backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1bidirectional/backward_simple_rnn/strided_slice_1?
=bidirectional/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
=bidirectional/backward_simple_rnn/TensorArrayV2/element_shape?
/bidirectional/backward_simple_rnn/TensorArrayV2TensorListReserveFbidirectional/backward_simple_rnn/TensorArrayV2/element_shape:output:0:bidirectional/backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/bidirectional/backward_simple_rnn/TensorArrayV2?
0bidirectional/backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 22
0bidirectional/backward_simple_rnn/ReverseV2/axis?
+bidirectional/backward_simple_rnn/ReverseV2	ReverseV2/bidirectional/backward_simple_rnn/transpose:y:09bidirectional/backward_simple_rnn/ReverseV2/axis:output:0*
T0*+
_output_shapes
:?????????2-
+bidirectional/backward_simple_rnn/ReverseV2?
Wbidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2Y
Wbidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
Ibidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor4bidirectional/backward_simple_rnn/ReverseV2:output:0`bidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02K
Ibidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
7bidirectional/backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7bidirectional/backward_simple_rnn/strided_slice_2/stack?
9bidirectional/backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_2/stack_1?
9bidirectional/backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_2/stack_2?
1bidirectional/backward_simple_rnn/strided_slice_2StridedSlice/bidirectional/backward_simple_rnn/transpose:y:0@bidirectional/backward_simple_rnn/strided_slice_2/stack:output:0Bbidirectional/backward_simple_rnn/strided_slice_2/stack_1:output:0Bbidirectional/backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask23
1bidirectional/backward_simple_rnn/strided_slice_2?
Ibidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpRbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02K
Ibidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
:bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul:bidirectional/backward_simple_rnn/strided_slice_2:output:0Qbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2<
:bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul?
Jbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpSbidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02L
Jbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
;bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAddDbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0Rbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2=
;bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
Kbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpTbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02M
Kbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
<bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMul0bidirectional/backward_simple_rnn/zeros:output:0Sbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2>
<bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
7bidirectional/backward_simple_rnn/simple_rnn_cell_2/addAddV2Dbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:0Fbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
29
7bidirectional/backward_simple_rnn/simple_rnn_cell_2/add?
8bidirectional/backward_simple_rnn/simple_rnn_cell_2/TanhTanh;bidirectional/backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2:
8bidirectional/backward_simple_rnn/simple_rnn_cell_2/Tanh?
?bidirectional/backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2A
?bidirectional/backward_simple_rnn/TensorArrayV2_1/element_shape?
1bidirectional/backward_simple_rnn/TensorArrayV2_1TensorListReserveHbidirectional/backward_simple_rnn/TensorArrayV2_1/element_shape:output:0:bidirectional/backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1bidirectional/backward_simple_rnn/TensorArrayV2_1?
&bidirectional/backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2(
&bidirectional/backward_simple_rnn/time?
:bidirectional/backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2<
:bidirectional/backward_simple_rnn/while/maximum_iterations?
4bidirectional/backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 26
4bidirectional/backward_simple_rnn/while/loop_counter?	
'bidirectional/backward_simple_rnn/whileWhile=bidirectional/backward_simple_rnn/while/loop_counter:output:0Cbidirectional/backward_simple_rnn/while/maximum_iterations:output:0/bidirectional/backward_simple_rnn/time:output:0:bidirectional/backward_simple_rnn/TensorArrayV2_1:handle:00bidirectional/backward_simple_rnn/zeros:output:0:bidirectional/backward_simple_rnn/strided_slice_1:output:0Ybidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Rbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceSbidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceTbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *?
body7R5
3bidirectional_backward_simple_rnn_while_body_381482*?
cond7R5
3bidirectional_backward_simple_rnn_while_cond_381481*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2)
'bidirectional/backward_simple_rnn/while?
Rbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2T
Rbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
Dbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack0bidirectional/backward_simple_rnn/while:output:3[bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02F
Dbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
7bidirectional/backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????29
7bidirectional/backward_simple_rnn/strided_slice_3/stack?
9bidirectional/backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9bidirectional/backward_simple_rnn/strided_slice_3/stack_1?
9bidirectional/backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_3/stack_2?
1bidirectional/backward_simple_rnn/strided_slice_3StridedSliceMbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional/backward_simple_rnn/strided_slice_3/stack:output:0Bbidirectional/backward_simple_rnn/strided_slice_3/stack_1:output:0Bbidirectional/backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask23
1bidirectional/backward_simple_rnn/strided_slice_3?
2bidirectional/backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2bidirectional/backward_simple_rnn/transpose_1/perm?
-bidirectional/backward_simple_rnn/transpose_1	TransposeMbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0;bidirectional/backward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2/
-bidirectional/backward_simple_rnn/transpose_1x
bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
bidirectional/concat/axis?
bidirectional/concatConcatV29bidirectional/forward_simple_rnn/strided_slice_3:output:0:bidirectional/backward_simple_rnn/strided_slice_3:output:0"bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
bidirectional/concat?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulbidirectional/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_1/dropout/Mul_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Reluc
dense_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
dense_3/Constg
dense_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_3/Const_1?
dense_3/clip_by_value/MinimumMinimumdense_3/Relu:activations:0dense_3/Const:output:0*
T0*'
_output_shapes
:?????????2
dense_3/clip_by_value/Minimum?
dense_3/clip_by_valueMaximum!dense_3/clip_by_value/Minimum:z:0dense_3/Const_1:output:0*
T0*'
_output_shapes
:?????????2
dense_3/clip_by_valuet
IdentityIdentitydense_3/clip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOpK^bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpJ^bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpL^bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp(^bidirectional/backward_simple_rnn/whileJ^bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpI^bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpK^bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp'^bidirectional/forward_simple_rnn/while^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 2?
Jbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpJbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Ibidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpIbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Kbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpKbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp2R
'bidirectional/backward_simple_rnn/while'bidirectional/backward_simple_rnn/while2?
Ibidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpIbidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
Hbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpHbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Jbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpJbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp2P
&bidirectional/forward_simple_rnn/while&bidirectional/forward_simple_rnn/while2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?K
?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383437
inputs_0B
0simple_rnn_cell_2_matmul_readvariableop_resource:
?
1simple_rnn_cell_2_biasadd_readvariableop_resource:
D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????2
	ReverseV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_383371*
condR
while_cond_383370*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_380408	
input
unknown
	unknown_0	
	unknown_1:
??
	unknown_2:

	unknown_3:

	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:
	unknown_9:

unknown_10:

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_3803792
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_nameInput:

_output_shapes
: 
?#
?
while_body_378934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_2_378956_0:
.
 while_simple_rnn_cell_2_378958_0:
2
 while_simple_rnn_cell_2_378960_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_2_378956:
,
while_simple_rnn_cell_2_378958:
0
while_simple_rnn_cell_2_378960:

??/while/simple_rnn_cell_2/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
/while/simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_2_378956_0 while_simple_rnn_cell_2_378958_0 while_simple_rnn_cell_2_378960_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_37892121
/while/simple_rnn_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp0^while/simple_rnn_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_2_378956 while_simple_rnn_cell_2_378956_0"B
while_simple_rnn_cell_2_378958 while_simple_rnn_cell_2_378958_0"B
while_simple_rnn_cell_2_378960 while_simple_rnn_cell_2_378960_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2b
/while/simple_rnn_cell_2/StatefulPartitionedCall/while/simple_rnn_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_382154
inputs_0U
Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
R
Dforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
W
Eforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

V
Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
S
Ebackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
X
Fbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?backward_simple_rnn/while?;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?forward_simple_rnn/whilel
forward_simple_rnn/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_simple_rnn/Shape?
&forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_simple_rnn/strided_slice/stack?
(forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_1?
(forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_2?
 forward_simple_rnn/strided_sliceStridedSlice!forward_simple_rnn/Shape:output:0/forward_simple_rnn/strided_slice/stack:output:01forward_simple_rnn/strided_slice/stack_1:output:01forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_simple_rnn/strided_slice?
forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2 
forward_simple_rnn/zeros/mul/y?
forward_simple_rnn/zeros/mulMul)forward_simple_rnn/strided_slice:output:0'forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/mul?
forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
forward_simple_rnn/zeros/Less/y?
forward_simple_rnn/zeros/LessLess forward_simple_rnn/zeros/mul:z:0(forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/Less?
!forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!forward_simple_rnn/zeros/packed/1?
forward_simple_rnn/zeros/packedPack)forward_simple_rnn/strided_slice:output:0*forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_simple_rnn/zeros/packed?
forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
forward_simple_rnn/zeros/Const?
forward_simple_rnn/zerosFill(forward_simple_rnn/zeros/packed:output:0'forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
forward_simple_rnn/zeros?
!forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_simple_rnn/transpose/perm?
forward_simple_rnn/transpose	Transposeinputs_0*forward_simple_rnn/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
forward_simple_rnn/transpose?
forward_simple_rnn/Shape_1Shape forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
forward_simple_rnn/Shape_1?
(forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_1/stack?
*forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_1?
*forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_2?
"forward_simple_rnn/strided_slice_1StridedSlice#forward_simple_rnn/Shape_1:output:01forward_simple_rnn/strided_slice_1/stack:output:03forward_simple_rnn/strided_slice_1/stack_1:output:03forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_1?
.forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.forward_simple_rnn/TensorArrayV2/element_shape?
 forward_simple_rnn/TensorArrayV2TensorListReserve7forward_simple_rnn/TensorArrayV2/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_simple_rnn/TensorArrayV2?
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????2J
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor forward_simple_rnn/transpose:y:0Qforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
(forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_2/stack?
*forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_1?
*forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_2?
"forward_simple_rnn/strided_slice_2StridedSlice forward_simple_rnn/transpose:y:01forward_simple_rnn/strided_slice_2/stack:output:03forward_simple_rnn/strided_slice_2/stack_1:output:03forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_2?
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpCforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02<
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
+forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul+forward_simple_rnn/strided_slice_2:output:0Bforward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2-
+forward_simple_rnn/simple_rnn_cell_1/MatMul?
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02=
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
,forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAdd5forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0Cforward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02>
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMul!forward_simple_rnn/zeros:output:0Dforward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
(forward_simple_rnn/simple_rnn_cell_1/addAddV25forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:07forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2*
(forward_simple_rnn/simple_rnn_cell_1/add?
)forward_simple_rnn/simple_rnn_cell_1/TanhTanh,forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2+
)forward_simple_rnn/simple_rnn_cell_1/Tanh?
0forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0forward_simple_rnn/TensorArrayV2_1/element_shape?
"forward_simple_rnn/TensorArrayV2_1TensorListReserve9forward_simple_rnn/TensorArrayV2_1/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"forward_simple_rnn/TensorArrayV2_1t
forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_simple_rnn/time?
+forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+forward_simple_rnn/while/maximum_iterations?
%forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%forward_simple_rnn/while/loop_counter?
forward_simple_rnn/whileWhile.forward_simple_rnn/while/loop_counter:output:04forward_simple_rnn/while/maximum_iterations:output:0 forward_simple_rnn/time:output:0+forward_simple_rnn/TensorArrayV2_1:handle:0!forward_simple_rnn/zeros:output:0+forward_simple_rnn/strided_slice_1:output:0Jforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourceDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourceEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$forward_simple_rnn_while_body_381976*0
cond(R&
$forward_simple_rnn_while_cond_381975*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
forward_simple_rnn/while?
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2E
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
5forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack!forward_simple_rnn/while:output:3Lforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype027
5forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
(forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(forward_simple_rnn/strided_slice_3/stack?
*forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*forward_simple_rnn/strided_slice_3/stack_1?
*forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_3/stack_2?
"forward_simple_rnn/strided_slice_3StridedSlice>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:01forward_simple_rnn/strided_slice_3/stack:output:03forward_simple_rnn/strided_slice_3/stack_1:output:03forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_3?
#forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#forward_simple_rnn/transpose_1/perm?
forward_simple_rnn/transpose_1	Transpose>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0,forward_simple_rnn/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2 
forward_simple_rnn/transpose_1n
backward_simple_rnn/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_simple_rnn/Shape?
'backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_simple_rnn/strided_slice/stack?
)backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_1?
)backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_2?
!backward_simple_rnn/strided_sliceStridedSlice"backward_simple_rnn/Shape:output:00backward_simple_rnn/strided_slice/stack:output:02backward_simple_rnn/strided_slice/stack_1:output:02backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_simple_rnn/strided_slice?
backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2!
backward_simple_rnn/zeros/mul/y?
backward_simple_rnn/zeros/mulMul*backward_simple_rnn/strided_slice:output:0(backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/zeros/mul?
 backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 backward_simple_rnn/zeros/Less/y?
backward_simple_rnn/zeros/LessLess!backward_simple_rnn/zeros/mul:z:0)backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_simple_rnn/zeros/Less?
"backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2$
"backward_simple_rnn/zeros/packed/1?
 backward_simple_rnn/zeros/packedPack*backward_simple_rnn/strided_slice:output:0+backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_simple_rnn/zeros/packed?
backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
backward_simple_rnn/zeros/Const?
backward_simple_rnn/zerosFill)backward_simple_rnn/zeros/packed:output:0(backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
backward_simple_rnn/zeros?
"backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_simple_rnn/transpose/perm?
backward_simple_rnn/transpose	Transposeinputs_0+backward_simple_rnn/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
backward_simple_rnn/transpose?
backward_simple_rnn/Shape_1Shape!backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
backward_simple_rnn/Shape_1?
)backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_1/stack?
+backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_1?
+backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_2?
#backward_simple_rnn/strided_slice_1StridedSlice$backward_simple_rnn/Shape_1:output:02backward_simple_rnn/strided_slice_1/stack:output:04backward_simple_rnn/strided_slice_1/stack_1:output:04backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_1?
/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/backward_simple_rnn/TensorArrayV2/element_shape?
!backward_simple_rnn/TensorArrayV2TensorListReserve8backward_simple_rnn/TensorArrayV2/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_simple_rnn/TensorArrayV2?
"backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2$
"backward_simple_rnn/ReverseV2/axis?
backward_simple_rnn/ReverseV2	ReverseV2!backward_simple_rnn/transpose:y:0+backward_simple_rnn/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
backward_simple_rnn/ReverseV2?
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????2K
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor&backward_simple_rnn/ReverseV2:output:0Rbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
)backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_2/stack?
+backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_1?
+backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_2?
#backward_simple_rnn/strided_slice_2StridedSlice!backward_simple_rnn/transpose:y:02backward_simple_rnn/strided_slice_2/stack:output:04backward_simple_rnn/strided_slice_2/stack_1:output:04backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_2?
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpDbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02=
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
,backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul,backward_simple_rnn/strided_slice_2:output:0Cbackward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,backward_simple_rnn/simple_rnn_cell_2/MatMul?
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02>
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
-backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd6backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0Dbackward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMul"backward_simple_rnn/zeros:output:0Ebackward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
20
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
)backward_simple_rnn/simple_rnn_cell_2/addAddV26backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:08backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2+
)backward_simple_rnn/simple_rnn_cell_2/add?
*backward_simple_rnn/simple_rnn_cell_2/TanhTanh-backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2,
*backward_simple_rnn/simple_rnn_cell_2/Tanh?
1backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   23
1backward_simple_rnn/TensorArrayV2_1/element_shape?
#backward_simple_rnn/TensorArrayV2_1TensorListReserve:backward_simple_rnn/TensorArrayV2_1/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#backward_simple_rnn/TensorArrayV2_1v
backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_simple_rnn/time?
,backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,backward_simple_rnn/while/maximum_iterations?
&backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_simple_rnn/while/loop_counter?
backward_simple_rnn/whileWhile/backward_simple_rnn/while/loop_counter:output:05backward_simple_rnn/while/maximum_iterations:output:0!backward_simple_rnn/time:output:0,backward_simple_rnn/TensorArrayV2_1:handle:0"backward_simple_rnn/zeros:output:0,backward_simple_rnn/strided_slice_1:output:0Kbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%backward_simple_rnn_while_body_382086*1
cond)R'
%backward_simple_rnn_while_cond_382085*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
backward_simple_rnn/while?
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2F
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
6backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack"backward_simple_rnn/while:output:3Mbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype028
6backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
)backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)backward_simple_rnn/strided_slice_3/stack?
+backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+backward_simple_rnn/strided_slice_3/stack_1?
+backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_3/stack_2?
#backward_simple_rnn/strided_slice_3StridedSlice?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:02backward_simple_rnn/strided_slice_3/stack:output:04backward_simple_rnn/strided_slice_3/stack_1:output:04backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_3?
$backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$backward_simple_rnn/transpose_1/perm?
backward_simple_rnn/transpose_1	Transpose?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0-backward_simple_rnn/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2!
backward_simple_rnn/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2+forward_simple_rnn/strided_slice_3:output:0,backward_simple_rnn/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp=^backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<^backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp>^backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp^backward_simple_rnn/while<^forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp=^forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp^forward_simple_rnn/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2|
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2z
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2~
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp26
backward_simple_rnn/whilebackward_simple_rnn/while2z
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp24
forward_simple_rnn/whileforward_simple_rnn/while:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
?
?
V2.3-RNN-Reg-Bidirect-Swivel-Balance_bidirectional_forward_simple_rnn_while_cond_378135?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_loop_counter?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_maximum_iterationsV
Rrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholderX
Trnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholder_1X
Trnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholder_2?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_less_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_strided_slice_1?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_cond_378135___redundant_placeholder0?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_cond_378135___redundant_placeholder1?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_cond_378135___redundant_placeholder2?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_cond_378135___redundant_placeholder3S
Ornn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity
?
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/LessLessRrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholder?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_less_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2Q
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Less?
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/IdentityIdentityS2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2U
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity"?
Ornn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?F
?
$forward_simple_rnn_while_body_380141B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2A
=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0}
yforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
Z
Lforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
_
Mforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

%
!forward_simple_rnn_while_identity'
#forward_simple_rnn_while_identity_1'
#forward_simple_rnn_while_identity_2'
#forward_simple_rnn_while_identity_3'
#forward_simple_rnn_while_identity_4?
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1{
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor[
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
X
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0$forward_simple_rnn_while_placeholderSforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02>
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02B
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
1forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulCforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Hforward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
23
1forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02C
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAdd;forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0Iforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02D
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMul&forward_simple_rnn_while_placeholder_2Jforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
.forward_simple_rnn/while/simple_rnn_cell_1/addAddV2;forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0=forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
20
.forward_simple_rnn/while/simple_rnn_cell_1/add?
/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanh2forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
21
/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&forward_simple_rnn_while_placeholder_1$forward_simple_rnn_while_placeholder3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_simple_rnn/while/add/y?
forward_simple_rnn/while/addAddV2$forward_simple_rnn_while_placeholder'forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/while/add?
 forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 forward_simple_rnn/while/add_1/y?
forward_simple_rnn/while/add_1AddV2>forward_simple_rnn_while_forward_simple_rnn_while_loop_counter)forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
forward_simple_rnn/while/add_1?
!forward_simple_rnn/while/IdentityIdentity"forward_simple_rnn/while/add_1:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity?
#forward_simple_rnn/while/Identity_1IdentityDforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_1?
#forward_simple_rnn/while/Identity_2Identity forward_simple_rnn/while/add:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_2?
#forward_simple_rnn/while/Identity_3IdentityMforward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_3?
#forward_simple_rnn/while/Identity_4Identity3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0^forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2%
#forward_simple_rnn/while/Identity_4?
forward_simple_rnn/while/NoOpNoOpB^forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpA^forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpC^forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_simple_rnn/while/NoOp"|
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0"S
#forward_simple_rnn_while_identity_1,forward_simple_rnn/while/Identity_1:output:0"S
#forward_simple_rnn_while_identity_2,forward_simple_rnn/while/Identity_2:output:0"S
#forward_simple_rnn_while_identity_3,forward_simple_rnn/while/Identity_3:output:0"S
#forward_simple_rnn_while_identity_4,forward_simple_rnn/while/Identity_4:output:0"?
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourceLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensoryforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpAforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpBforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_forward_simple_rnn_layer_call_fn_382706

inputs
unknown:

	unknown_0:

	unknown_1:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_3795272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?I
?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_383165

inputsB
0simple_rnn_cell_1_matmul_readvariableop_resource:
?
1simple_rnn_cell_1_biasadd_readvariableop_resource:
D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_1/BiasAdd/ReadVariableOp?'simple_rnn_cell_1/MatMul/ReadVariableOp?)simple_rnn_cell_1/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOp?
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul?
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOp?
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/BiasAdd?
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOp?
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul_1?
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/add?
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_383099*
condR
while_cond_383098*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_embed_layer_call_and_return_conditional_losses_380073

inputs
unknown
	unknown_0	
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_2217052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
c
*__inference_dropout_1_layer_call_fn_382632

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3804382
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
? 
"__inference__traced_restore_384108
file_prefix-
assignvariableop_variable:
??3
!assignvariableop_1_dense_2_kernel:-
assignvariableop_2_dense_2_bias:3
!assignvariableop_3_dense_3_kernel:-
assignvariableop_4_dense_3_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: _
Massignvariableop_10_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel:
i
Wassignvariableop_11_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel:

Y
Kassignvariableop_12_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias:
`
Nassignvariableop_13_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel:
j
Xassignvariableop_14_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel:

Z
Lassignvariableop_15_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias:
#
assignvariableop_16_total: #
assignvariableop_17_count: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: %
assignvariableop_20_total_2: %
assignvariableop_21_count_2: 7
#assignvariableop_22_adam_variable_m:
??;
)assignvariableop_23_adam_dense_2_kernel_m:5
'assignvariableop_24_adam_dense_2_bias_m:;
)assignvariableop_25_adam_dense_3_kernel_m:5
'assignvariableop_26_adam_dense_3_bias_m:f
Tassignvariableop_27_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_m:
p
^assignvariableop_28_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_m:

`
Rassignvariableop_29_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_m:
g
Uassignvariableop_30_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_m:
q
_assignvariableop_31_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_m:

a
Sassignvariableop_32_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_m:
7
#assignvariableop_33_adam_variable_v:
??;
)assignvariableop_34_adam_dense_2_kernel_v:5
'assignvariableop_35_adam_dense_2_bias_v:;
)assignvariableop_36_adam_dense_3_kernel_v:5
'assignvariableop_37_adam_dense_3_bias_v:f
Tassignvariableop_38_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_v:
p
^assignvariableop_39_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_v:

`
Rassignvariableop_40_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_v:
g
Uassignvariableop_41_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_v:
q
_assignvariableop_42_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_v:

a
Sassignvariableop_43_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_v:

identity_45??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*?
value?B?-B:layer_with_weights-0/Variable:0/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/Variable:0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/Variable:0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_3_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpMassignvariableop_10_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpWassignvariableop_11_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpKassignvariableop_12_bidirectional_forward_simple_rnn_simple_rnn_cell_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpNassignvariableop_13_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpXassignvariableop_14_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpLassignvariableop_15_bidirectional_backward_simple_rnn_simple_rnn_cell_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_adam_variable_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpTassignvariableop_27_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp^assignvariableop_28_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpRassignvariableop_29_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpUassignvariableop_30_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp_assignvariableop_31_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpSassignvariableop_32_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp#assignvariableop_33_adam_variable_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_2_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_2_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_3_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_3_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpTassignvariableop_38_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp^assignvariableop_39_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpRassignvariableop_40_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpUassignvariableop_41_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp_assignvariableop_42_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpSassignvariableop_43_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_439
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_44f
Identity_45IdentityIdentity_44:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_45?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_45Identity_45:output:0*m
_input_shapes\
Z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?V
?
3bidirectional_backward_simple_rnn_while_body_381223`
\bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_loop_counterf
bbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_maximum_iterations7
3bidirectional_backward_simple_rnn_while_placeholder9
5bidirectional_backward_simple_rnn_while_placeholder_19
5bidirectional_backward_simple_rnn_while_placeholder_2_
[bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_strided_slice_1_0?
?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0l
Zbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
i
[bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
n
\bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

4
0bidirectional_backward_simple_rnn_while_identity6
2bidirectional_backward_simple_rnn_while_identity_16
2bidirectional_backward_simple_rnn_while_identity_26
2bidirectional_backward_simple_rnn_while_identity_36
2bidirectional_backward_simple_rnn_while_identity_4]
Ybidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_strided_slice_1?
?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensorj
Xbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
g
Ybidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
l
Zbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??Pbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Obidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Qbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Ybidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2[
Ybidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Kbidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_03bidirectional_backward_simple_rnn_while_placeholderbbidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02M
Kbidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Obidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpZbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02Q
Obidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
@bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulRbidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Wbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2B
@bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
Pbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp[bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02R
Pbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
Abidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAddJbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Xbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2C
Abidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Qbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp\bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02S
Qbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Bbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul5bidirectional_backward_simple_rnn_while_placeholder_2Ybidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2D
Bbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
=bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2Jbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0Lbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2?
=bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/add?
>bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/TanhTanhAbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2@
>bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
Lbidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem5bidirectional_backward_simple_rnn_while_placeholder_13bidirectional_backward_simple_rnn_while_placeholderBbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02N
Lbidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
-bidirectional/backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-bidirectional/backward_simple_rnn/while/add/y?
+bidirectional/backward_simple_rnn/while/addAddV23bidirectional_backward_simple_rnn_while_placeholder6bidirectional/backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2-
+bidirectional/backward_simple_rnn/while/add?
/bidirectional/backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :21
/bidirectional/backward_simple_rnn/while/add_1/y?
-bidirectional/backward_simple_rnn/while/add_1AddV2\bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_loop_counter8bidirectional/backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2/
-bidirectional/backward_simple_rnn/while/add_1?
0bidirectional/backward_simple_rnn/while/IdentityIdentity1bidirectional/backward_simple_rnn/while/add_1:z:0-^bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 22
0bidirectional/backward_simple_rnn/while/Identity?
2bidirectional/backward_simple_rnn/while/Identity_1Identitybbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_maximum_iterations-^bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 24
2bidirectional/backward_simple_rnn/while/Identity_1?
2bidirectional/backward_simple_rnn/while/Identity_2Identity/bidirectional/backward_simple_rnn/while/add:z:0-^bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 24
2bidirectional/backward_simple_rnn/while/Identity_2?
2bidirectional/backward_simple_rnn/while/Identity_3Identity\bidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 24
2bidirectional/backward_simple_rnn/while/Identity_3?
2bidirectional/backward_simple_rnn/while/Identity_4IdentityBbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0-^bidirectional/backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
24
2bidirectional/backward_simple_rnn/while/Identity_4?
,bidirectional/backward_simple_rnn/while/NoOpNoOpQ^bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpP^bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpR^bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2.
,bidirectional/backward_simple_rnn/while/NoOp"?
Ybidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_strided_slice_1[bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_strided_slice_1_0"m
0bidirectional_backward_simple_rnn_while_identity9bidirectional/backward_simple_rnn/while/Identity:output:0"q
2bidirectional_backward_simple_rnn_while_identity_1;bidirectional/backward_simple_rnn/while/Identity_1:output:0"q
2bidirectional_backward_simple_rnn_while_identity_2;bidirectional/backward_simple_rnn/while/Identity_2:output:0"q
2bidirectional_backward_simple_rnn_while_identity_3;bidirectional/backward_simple_rnn/while/Identity_3:output:0"q
2bidirectional_backward_simple_rnn_while_identity_4;bidirectional/backward_simple_rnn/while/Identity_4:output:0"?
Ybidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource[bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Zbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource\bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Xbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceZbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Pbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpPbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Obidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpObidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Qbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpQbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?F
?
$forward_simple_rnn_while_body_380516B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2A
=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0}
yforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
Z
Lforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
_
Mforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

%
!forward_simple_rnn_while_identity'
#forward_simple_rnn_while_identity_1'
#forward_simple_rnn_while_identity_2'
#forward_simple_rnn_while_identity_3'
#forward_simple_rnn_while_identity_4?
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1{
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor[
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
X
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0$forward_simple_rnn_while_placeholderSforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02>
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02B
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
1forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulCforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Hforward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
23
1forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02C
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAdd;forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0Iforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02D
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMul&forward_simple_rnn_while_placeholder_2Jforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
.forward_simple_rnn/while/simple_rnn_cell_1/addAddV2;forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0=forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
20
.forward_simple_rnn/while/simple_rnn_cell_1/add?
/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanh2forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
21
/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&forward_simple_rnn_while_placeholder_1$forward_simple_rnn_while_placeholder3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_simple_rnn/while/add/y?
forward_simple_rnn/while/addAddV2$forward_simple_rnn_while_placeholder'forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/while/add?
 forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 forward_simple_rnn/while/add_1/y?
forward_simple_rnn/while/add_1AddV2>forward_simple_rnn_while_forward_simple_rnn_while_loop_counter)forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
forward_simple_rnn/while/add_1?
!forward_simple_rnn/while/IdentityIdentity"forward_simple_rnn/while/add_1:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity?
#forward_simple_rnn/while/Identity_1IdentityDforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_1?
#forward_simple_rnn/while/Identity_2Identity forward_simple_rnn/while/add:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_2?
#forward_simple_rnn/while/Identity_3IdentityMforward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_3?
#forward_simple_rnn/while/Identity_4Identity3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0^forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2%
#forward_simple_rnn/while/Identity_4?
forward_simple_rnn/while/NoOpNoOpB^forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpA^forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpC^forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_simple_rnn/while/NoOp"|
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0"S
#forward_simple_rnn_while_identity_1,forward_simple_rnn/while/Identity_1:output:0"S
#forward_simple_rnn_while_identity_2,forward_simple_rnn/while/Identity_2:output:0"S
#forward_simple_rnn_while_identity_3,forward_simple_rnn/while/Identity_3:output:0"S
#forward_simple_rnn_while_identity_4,forward_simple_rnn/while/Identity_4:output:0"?
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourceLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensoryforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpAforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpBforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
D
(__inference_reshape_layer_call_fn_381625

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3800942
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$forward_simple_rnn_while_cond_382199B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2D
@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_382199___redundant_placeholder0Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_382199___redundant_placeholder1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_382199___redundant_placeholder2Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_382199___redundant_placeholder3%
!forward_simple_rnn_while_identity
?
forward_simple_rnn/while/LessLess$forward_simple_rnn_while_placeholder@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
forward_simple_rnn/while/Less?
!forward_simple_rnn/while/IdentityIdentity!forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
C__inference_dense_2_layer_call_and_return_conditional_losses_382622

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
__inference__preprocess_92
	sentences:
6none_lookup_none_lookup_lookuptablefindv2_table_handle;
7none_lookup_none_lookup_lookuptablefindv2_default_value	

identity_3	

identity_4	

identity_5	??)None_Lookup/None_Lookup/LookupTableFindV2?'None_Lookup/None_Size/LookupTableSizeV2?+None_Lookup_1/None_Lookup/LookupTableFindV2?)None_Lookup_1/None_Size/LookupTableSizeV2?
normalize/StaticRegexReplaceStaticRegexReplace	sentences*
_output_shapes
:*
pattern	[\p{P}]*
rewrite 2
normalize/StaticRegexReplacek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDims%normalize/StaticRegexReplace:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:2

ExpandDims?
"DenseToSparseTensor/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 2$
"DenseToSparseTensor/ignore_value/x?
DenseToSparseTensor/NotEqualNotEqualExpandDims:output:0+DenseToSparseTensor/ignore_value/x:output:0*
T0*
_output_shapes
:2
DenseToSparseTensor/NotEqual?
DenseToSparseTensor/indicesWhere DenseToSparseTensor/NotEqual:z:0*0
_output_shapes
:??????????????????2
DenseToSparseTensor/indices?
DenseToSparseTensor/valuesGatherNdExpandDims:output:0#DenseToSparseTensor/indices:index:0*
Tindices0	*
Tparams0*
_output_shapes
:2
DenseToSparseTensor/values?
DenseToSparseTensor/dense_shapeShapeExpandDims:output:0*
T0*#
_output_shapes
:?????????*
out_type0	2!
DenseToSparseTensor/dense_shapeb
tokenize/ConstConst*
_output_shapes
: *
dtype0*
value	B B 2
tokenize/Const?
tokenize/StringSplitStringSplit%normalize/StaticRegexReplace:output:0tokenize/Const:output:0*<
_output_shapes*
(:?????????:?????????:2
tokenize/StringSplit?
None_Lookup/hash_bucketStringToHashBucketFasttokenize/StringSplit:values:0*#
_output_shapes
:?????????*
num_bucketsk2
None_Lookup/hash_bucket?
)None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV26none_lookup_none_lookup_lookuptablefindv2_table_handletokenize/StringSplit:values:07none_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*
_output_shapes
:2+
)None_Lookup/None_Lookup/LookupTableFindV2?
'None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV26none_lookup_none_lookup_lookuptablefindv2_table_handle*^None_Lookup/None_Lookup/LookupTableFindV2*
_output_shapes
: 2)
'None_Lookup/None_Size/LookupTableSizeV2?
None_Lookup/AddAdd None_Lookup/hash_bucket:output:0.None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*#
_output_shapes
:?????????2
None_Lookup/Add?
None_Lookup/NotEqualNotEqual2None_Lookup/None_Lookup/LookupTableFindV2:values:07none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*
_output_shapes
:2
None_Lookup/NotEqual?
None_Lookup/SelectSelectNone_Lookup/NotEqual:z:02None_Lookup/None_Lookup/LookupTableFindV2:values:0None_Lookup/Add:z:0*
T0	*#
_output_shapes
:?????????2
None_Lookup/SelectO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?
None_Lookup_1/hash_bucketStringToHashBucketFastConst:output:0*
_output_shapes
: *
num_bucketsk2
None_Lookup_1/hash_bucket?
+None_Lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV26none_lookup_none_lookup_lookuptablefindv2_table_handleConst:output:07none_lookup_none_lookup_lookuptablefindv2_default_value(^None_Lookup/None_Size/LookupTableSizeV2*	
Tin0*

Tout0	*
_output_shapes
:2-
+None_Lookup_1/None_Lookup/LookupTableFindV2?
)None_Lookup_1/None_Size/LookupTableSizeV2LookupTableSizeV26none_lookup_none_lookup_lookuptablefindv2_table_handle,^None_Lookup_1/None_Lookup/LookupTableFindV2*
_output_shapes
: 2+
)None_Lookup_1/None_Size/LookupTableSizeV2?
None_Lookup_1/AddAdd"None_Lookup_1/hash_bucket:output:00None_Lookup_1/None_Size/LookupTableSizeV2:size:0*
T0	*
_output_shapes
: 2
None_Lookup_1/Add?
None_Lookup_1/NotEqualNotEqual4None_Lookup_1/None_Lookup/LookupTableFindV2:values:07none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*
_output_shapes
:2
None_Lookup_1/NotEqual?
None_Lookup_1/SelectSelectNone_Lookup_1/NotEqual:z:04None_Lookup_1/None_Lookup/LookupTableFindV2:values:0None_Lookup_1/Add:z:0*
T0	*
_output_shapes
: 2
None_Lookup_1/Select?
'SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowstokenize/StringSplit:indices:0None_Lookup/Select:output:0tokenize/StringSplit:shape:0None_Lookup_1/Select:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????2)
'SparseFillEmptyRows/SparseFillEmptyRows?
IdentityIdentity8SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0*
T0	*'
_output_shapes
:?????????2

Identity?

Identity_1Identity7SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*#
_output_shapes
:?????????2

Identity_1g

Identity_2Identitytokenize/StringSplit:shape:0*
T0	*
_output_shapes
:2

Identity_2p
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Max/reduction_indicesi
MaxMaxIdentity:output:0Max/reduction_indices:output:0*
T0	*
_output_shapes
:2
MaxT
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Const_1l
ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
ones_like/Shaped
ones_like/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R2
ones_like/Constw
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0	*
_output_shapes
:2
	ones_likeX
AddAddMax:output:0ones_like:output:0*
T0	*
_output_shapes
:2
Add]
MaximumMaximumConst_1:output:0Add:z:0*
T0	*
_output_shapes
:2	
Maximum?

Identity_3IdentityIdentity:output:0*^None_Lookup/None_Lookup/LookupTableFindV2(^None_Lookup/None_Size/LookupTableSizeV2,^None_Lookup_1/None_Lookup/LookupTableFindV2*^None_Lookup_1/None_Size/LookupTableSizeV2*
T0	*'
_output_shapes
:?????????2

Identity_3?

Identity_4IdentityIdentity_1:output:0*^None_Lookup/None_Lookup/LookupTableFindV2(^None_Lookup/None_Size/LookupTableSizeV2,^None_Lookup_1/None_Lookup/LookupTableFindV2*^None_Lookup_1/None_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????2

Identity_4?

Identity_5IdentityMaximum:z:0*^None_Lookup/None_Lookup/LookupTableFindV2(^None_Lookup/None_Size/LookupTableSizeV2,^None_Lookup_1/None_Lookup/LookupTableFindV2*^None_Lookup_1/None_Size/LookupTableSizeV2*
T0	*
_output_shapes
:2

Identity_5"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:::2V
)None_Lookup/None_Lookup/LookupTableFindV2)None_Lookup/None_Lookup/LookupTableFindV22R
'None_Lookup/None_Size/LookupTableSizeV2'None_Lookup/None_Size/LookupTableSizeV22Z
+None_Lookup_1/None_Lookup/LookupTableFindV2+None_Lookup_1/None_Lookup/LookupTableFindV22V
)None_Lookup_1/None_Size/LookupTableSizeV2)None_Lookup_1/None_Size/LookupTableSizeV2: 

_output_shapes
::

_output_shapes
:
?!
?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_380818

inputs
embed_380785
embed_380787	 
embed_380789:
??&
bidirectional_380793:
"
bidirectional_380795:
&
bidirectional_380797:

&
bidirectional_380799:
"
bidirectional_380801:
&
bidirectional_380803:

 
dense_2_380806:
dense_2_380808: 
dense_3_380812:
dense_3_380814:
identity??%bidirectional/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?embed/StatefulPartitionedCall?
embed/StatefulPartitionedCallStatefulPartitionedCallinputsembed_380785embed_380787embed_380789*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_embed_layer_call_and_return_conditional_losses_3807402
embed/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall&embed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3800942
reshape/PartitionedCall?
%bidirectional/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0bidirectional_380793bidirectional_380795bidirectional_380797bidirectional_380799bidirectional_380801bidirectional_380803*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_bidirectional_layer_call_and_return_conditional_losses_3806942'
%bidirectional/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_2_380806dense_2_380808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3803442!
dense_2/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3804382#
!dropout_1/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_380812dense_3_380814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3803722!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^bidirectional/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^embed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?L
?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_379648

inputsB
0simple_rnn_cell_2_matmul_readvariableop_resource:
?
1simple_rnn_cell_2_biasadd_readvariableop_resource:
D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	ReverseV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_379582*
condR
while_cond_379581*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
while_cond_379581
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_379581___redundant_placeholder04
0while_while_cond_379581___redundant_placeholder14
0while_while_cond_379581___redundant_placeholder24
0while_while_cond_379581___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?F
?
$forward_simple_rnn_while_body_381752B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2A
=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0}
yforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
Z
Lforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
_
Mforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

%
!forward_simple_rnn_while_identity'
#forward_simple_rnn_while_identity_1'
#forward_simple_rnn_while_identity_2'
#forward_simple_rnn_while_identity_3'
#forward_simple_rnn_while_identity_4?
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1{
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor[
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
X
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????2L
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0$forward_simple_rnn_while_placeholderSforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02>
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02B
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
1forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulCforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Hforward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
23
1forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02C
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAdd;forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0Iforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02D
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMul&forward_simple_rnn_while_placeholder_2Jforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
.forward_simple_rnn/while/simple_rnn_cell_1/addAddV2;forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0=forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
20
.forward_simple_rnn/while/simple_rnn_cell_1/add?
/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanh2forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
21
/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&forward_simple_rnn_while_placeholder_1$forward_simple_rnn_while_placeholder3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_simple_rnn/while/add/y?
forward_simple_rnn/while/addAddV2$forward_simple_rnn_while_placeholder'forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/while/add?
 forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 forward_simple_rnn/while/add_1/y?
forward_simple_rnn/while/add_1AddV2>forward_simple_rnn_while_forward_simple_rnn_while_loop_counter)forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
forward_simple_rnn/while/add_1?
!forward_simple_rnn/while/IdentityIdentity"forward_simple_rnn/while/add_1:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity?
#forward_simple_rnn/while/Identity_1IdentityDforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_1?
#forward_simple_rnn/while/Identity_2Identity forward_simple_rnn/while/add:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_2?
#forward_simple_rnn/while/Identity_3IdentityMforward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_3?
#forward_simple_rnn/while/Identity_4Identity3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0^forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2%
#forward_simple_rnn/while/Identity_4?
forward_simple_rnn/while/NoOpNoOpB^forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpA^forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpC^forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_simple_rnn/while/NoOp"|
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0"S
#forward_simple_rnn_while_identity_1,forward_simple_rnn/while/Identity_1:output:0"S
#forward_simple_rnn_while_identity_2,forward_simple_rnn/while/Identity_2:output:0"S
#forward_simple_rnn_while_identity_3,forward_simple_rnn/while/Identity_3:output:0"S
#forward_simple_rnn_while_identity_4,forward_simple_rnn/while/Identity_4:output:0"?
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourceLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensoryforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpAforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpBforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?L
?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383551

inputsB
0simple_rnn_cell_2_matmul_readvariableop_resource:
?
1simple_rnn_cell_2_biasadd_readvariableop_resource:
D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	ReverseV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_383485*
condR
while_cond_383484*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_378921

inputs

states0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????
2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityg

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_namestates
?
?
%backward_simple_rnn_while_cond_382085D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2F
Bbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382085___redundant_placeholder0\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382085___redundant_placeholder1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382085___redundant_placeholder2\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382085___redundant_placeholder3&
"backward_simple_rnn_while_identity
?
backward_simple_rnn/while/LessLess%backward_simple_rnn_while_placeholderBbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2 
backward_simple_rnn/while/Less?
"backward_simple_rnn/while/IdentityIdentity"backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?#
?
while_body_378572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_1_378594_0:
.
 while_simple_rnn_cell_1_378596_0:
2
 while_simple_rnn_cell_1_378598_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_1_378594:
,
while_simple_rnn_cell_1_378596:
0
while_simple_rnn_cell_1_378598:

??/while/simple_rnn_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
/while/simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_1_378594_0 while_simple_rnn_cell_1_378596_0 while_simple_rnn_cell_1_378598_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_37850521
/while/simple_rnn_cell_1/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp0^while/simple_rnn_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_1_378594 while_simple_rnn_cell_1_378594_0"B
while_simple_rnn_cell_1_378596 while_simple_rnn_cell_1_378596_0"B
while_simple_rnn_cell_1_378598 while_simple_rnn_cell_1_378598_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2b
/while/simple_rnn_cell_1/StatefulPartitionedCall/while/simple_rnn_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_forward_simple_rnn_layer_call_fn_382717

inputs
unknown:

	unknown_0:

	unknown_1:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_3799352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?|
?
W2.3-RNN-Reg-Bidirect-Swivel-Balance_bidirectional_backward_simple_rnn_while_body_378246?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_loop_counter?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_maximum_iterationsW
Srnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholderY
Urnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholder_1Y
Urnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholder_2?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_strided_slice_1_0?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0?
zrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
?
{rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
?
|rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

T
Prnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identityV
Rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity_1V
Rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity_2V
Rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity_3V
Rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity_4?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_strided_slice_1?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor?
xrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
?
yrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
?
zrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?s2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?u2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
}2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
}2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0Srnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholder?2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02q
o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
s2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpzrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02u
s2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulv2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0{2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2f
d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp{rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02v
t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAddn2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0|2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2g
e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
u2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp|rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02w
u2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMulUrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholder_2}2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2h
f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
a2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0p2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2c
a2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/add?
b2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/TanhTanhe2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2d
b2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
p2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemUrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholder_1Srnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholderf2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02r
p2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2S
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add/y?
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/addAddV2Srnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholderZ2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2Q
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add?
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2U
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add_1/y?
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add_1AddV2?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_loop_counter\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2S
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add_1?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/IdentityIdentityU2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add_1:z:0Q^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity?
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_1Identity?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_maximum_iterationsQ^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2X
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_1?
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_2IdentityS2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/add:z:0Q^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2X
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_2?
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_3Identity?2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0Q^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2X
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_3?
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_4Identityf2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0Q^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2X
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_4?
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/NoOpNoOpu^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpt^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpv^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2R
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/NoOp"?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_strided_slice_1?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_strided_slice_1_0"?
Prnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity:output:0"?
Rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity_1_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_1:output:0"?
Rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity_2_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_2:output:0"?
Rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity_3_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_3:output:0"?
Rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity_4_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity_4:output:0"?
yrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource{rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
zrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource|rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
xrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourcezrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpt2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
s2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOps2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
u2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpu2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?"
?
__inference___call___115
	sentences"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2	Y
Eembedding_lookup_sparse_embedding_lookup_read_readvariableop_resource:
??
identity??StatefulPartitionedCall?(embedding_lookup_sparse/embedding_lookup?<embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp?
StatefulPartitionedCallStatefulPartitionedCall	sentencesstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2	*
Tout
2			*)
_gradient_op_typePartitionedCall-92* 
_output_shapes
:::*#
fR
__inference__preprocess_922
StatefulPartitionedCall?
+embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+embedding_lookup_sparse/strided_slice/stack?
-embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-embedding_lookup_sparse/strided_slice/stack_1?
-embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-embedding_lookup_sparse/strided_slice/stack_2?
%embedding_lookup_sparse/strided_sliceStridedSlice StatefulPartitionedCall:output:04embedding_lookup_sparse/strided_slice/stack:output:06embedding_lookup_sparse/strided_slice/stack_1:output:06embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2'
%embedding_lookup_sparse/strided_slice?
embedding_lookup_sparse/CastCast.embedding_lookup_sparse/strided_slice:output:0*

DstT0*

SrcT0	*
_output_shapes
:2
embedding_lookup_sparse/Cast?
embedding_lookup_sparse/UniqueUnique StatefulPartitionedCall:output:1*
T0	*'
_output_shapes
:?????????:2 
embedding_lookup_sparse/Unique?
<embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOpReadVariableOpEembedding_lookup_sparse_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype02>
<embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp?
1embedding_lookup_sparse/embedding_lookup/IdentityIdentityDembedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:23
1embedding_lookup_sparse/embedding_lookup/Identity?
(embedding_lookup_sparse/embedding_lookupResourceGatherEembedding_lookup_sparse_embedding_lookup_read_readvariableop_resource"embedding_lookup_sparse/Unique:y:0=^embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*O
_classE
CAloc:@embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*
_output_shapes
:*
dtype02*
(embedding_lookup_sparse/embedding_lookup?
3embedding_lookup_sparse/embedding_lookup/Identity_1Identity1embedding_lookup_sparse/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*O
_classE
CAloc:@embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*
_output_shapes
:25
3embedding_lookup_sparse/embedding_lookup/Identity_1?
3embedding_lookup_sparse/embedding_lookup/Identity_2Identity<embedding_lookup_sparse/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:25
3embedding_lookup_sparse/embedding_lookup/Identity_2?
embedding_lookup_sparseSparseSegmentSqrtN<embedding_lookup_sparse/embedding_lookup/Identity_2:output:0$embedding_lookup_sparse/Unique:idx:0 embedding_lookup_sparse/Cast:y:0*
T0*
_output_shapes
:2
embedding_lookup_sparse?
IdentityIdentity embedding_lookup_sparse:output:0^StatefulPartitionedCall)^embedding_lookup_sparse/embedding_lookup=^embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
::::22
StatefulPartitionedCallStatefulPartitionedCall2T
(embedding_lookup_sparse/embedding_lookup(embedding_lookup_sparse/embedding_lookup2|
<embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp<embedding_lookup_sparse/embedding_lookup/Read/ReadVariableOp: 

_output_shapes
::

_output_shapes
:
?1
?
while_body_379869
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_1/MatMul/ReadVariableOp?/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOp?
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_1/MatMul?
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_1/BiasAdd?
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_1/MatMul_1?
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/add?
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
4__inference_backward_simple_rnn_layer_call_fn_383198

inputs
unknown:

	unknown_0:

	unknown_1:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_3796482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?I
?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_382829
inputs_0B
0simple_rnn_cell_1_matmul_readvariableop_resource:
?
1simple_rnn_cell_1_biasadd_readvariableop_resource:
D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_1/BiasAdd/ReadVariableOp?'simple_rnn_cell_1/MatMul/ReadVariableOp?)simple_rnn_cell_1/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOp?
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul?
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOp?
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/BiasAdd?
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOp?
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul_1?
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/add?
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_382763*
condR
while_cond_382762*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_383256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_383256___redundant_placeholder04
0while_while_cond_383256___redundant_placeholder14
0while_while_cond_383256___redundant_placeholder24
0while_while_cond_383256___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
$forward_simple_rnn_while_cond_381751B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2D
@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_381751___redundant_placeholder0Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_381751___redundant_placeholder1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_381751___redundant_placeholder2Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_381751___redundant_placeholder3%
!forward_simple_rnn_while_identity
?
forward_simple_rnn/while/LessLess$forward_simple_rnn_while_placeholder@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
forward_simple_rnn/while/Less?
!forward_simple_rnn/while/IdentityIdentity!forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
$forward_simple_rnn_while_cond_382423B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2D
@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_382423___redundant_placeholder0Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_382423___redundant_placeholder1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_382423___redundant_placeholder2Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_382423___redundant_placeholder3%
!forward_simple_rnn_while_identity
?
forward_simple_rnn/while/LessLess$forward_simple_rnn_while_placeholder@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
forward_simple_rnn/while/Less?
!forward_simple_rnn/while/IdentityIdentity!forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
__inference__initializer_134
table_init_asset_filepath9
5table_init_initializetablefromtextfilev2_table_handle
identity??(table_init/InitializeTableFromTextFileV2?
(table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV25table_init_initializetablefromtextfilev2_table_handletable_init_asset_filepath*
_output_shapes
 *
	key_index?????????*
value_index?????????2*
(table_init/InitializeTableFromTextFileV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const|
IdentityIdentityConst:output:0)^table_init/InitializeTableFromTextFileV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::2T
(table_init/InitializeTableFromTextFileV2(table_init/InitializeTableFromTextFileV2: 

_output_shapes
:
?
?
$__inference_signature_wrapper_380989	
input
unknown
	unknown_0	
	unknown_1:
??
	unknown_2:

	unknown_3:

	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:
	unknown_9:

unknown_10:

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3783332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_nameInput:

_output_shapes
: 
?
?
&__inference_embed_layer_call_fn_381598

inputs
unknown
	unknown_0	
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_embed_layer_call_and_return_conditional_losses_3807402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
??
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_381930
inputs_0U
Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
R
Dforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
W
Eforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

V
Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
S
Ebackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
X
Fbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?backward_simple_rnn/while?;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?forward_simple_rnn/whilel
forward_simple_rnn/ShapeShapeinputs_0*
T0*
_output_shapes
:2
forward_simple_rnn/Shape?
&forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_simple_rnn/strided_slice/stack?
(forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_1?
(forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_2?
 forward_simple_rnn/strided_sliceStridedSlice!forward_simple_rnn/Shape:output:0/forward_simple_rnn/strided_slice/stack:output:01forward_simple_rnn/strided_slice/stack_1:output:01forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_simple_rnn/strided_slice?
forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2 
forward_simple_rnn/zeros/mul/y?
forward_simple_rnn/zeros/mulMul)forward_simple_rnn/strided_slice:output:0'forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/mul?
forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
forward_simple_rnn/zeros/Less/y?
forward_simple_rnn/zeros/LessLess forward_simple_rnn/zeros/mul:z:0(forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/Less?
!forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!forward_simple_rnn/zeros/packed/1?
forward_simple_rnn/zeros/packedPack)forward_simple_rnn/strided_slice:output:0*forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_simple_rnn/zeros/packed?
forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
forward_simple_rnn/zeros/Const?
forward_simple_rnn/zerosFill(forward_simple_rnn/zeros/packed:output:0'forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
forward_simple_rnn/zeros?
!forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_simple_rnn/transpose/perm?
forward_simple_rnn/transpose	Transposeinputs_0*forward_simple_rnn/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
forward_simple_rnn/transpose?
forward_simple_rnn/Shape_1Shape forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
forward_simple_rnn/Shape_1?
(forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_1/stack?
*forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_1?
*forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_2?
"forward_simple_rnn/strided_slice_1StridedSlice#forward_simple_rnn/Shape_1:output:01forward_simple_rnn/strided_slice_1/stack:output:03forward_simple_rnn/strided_slice_1/stack_1:output:03forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_1?
.forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.forward_simple_rnn/TensorArrayV2/element_shape?
 forward_simple_rnn/TensorArrayV2TensorListReserve7forward_simple_rnn/TensorArrayV2/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_simple_rnn/TensorArrayV2?
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????2J
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor forward_simple_rnn/transpose:y:0Qforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
(forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_2/stack?
*forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_1?
*forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_2?
"forward_simple_rnn/strided_slice_2StridedSlice forward_simple_rnn/transpose:y:01forward_simple_rnn/strided_slice_2/stack:output:03forward_simple_rnn/strided_slice_2/stack_1:output:03forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_2?
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpCforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02<
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
+forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul+forward_simple_rnn/strided_slice_2:output:0Bforward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2-
+forward_simple_rnn/simple_rnn_cell_1/MatMul?
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02=
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
,forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAdd5forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0Cforward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02>
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMul!forward_simple_rnn/zeros:output:0Dforward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
(forward_simple_rnn/simple_rnn_cell_1/addAddV25forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:07forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2*
(forward_simple_rnn/simple_rnn_cell_1/add?
)forward_simple_rnn/simple_rnn_cell_1/TanhTanh,forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2+
)forward_simple_rnn/simple_rnn_cell_1/Tanh?
0forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0forward_simple_rnn/TensorArrayV2_1/element_shape?
"forward_simple_rnn/TensorArrayV2_1TensorListReserve9forward_simple_rnn/TensorArrayV2_1/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"forward_simple_rnn/TensorArrayV2_1t
forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_simple_rnn/time?
+forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+forward_simple_rnn/while/maximum_iterations?
%forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%forward_simple_rnn/while/loop_counter?
forward_simple_rnn/whileWhile.forward_simple_rnn/while/loop_counter:output:04forward_simple_rnn/while/maximum_iterations:output:0 forward_simple_rnn/time:output:0+forward_simple_rnn/TensorArrayV2_1:handle:0!forward_simple_rnn/zeros:output:0+forward_simple_rnn/strided_slice_1:output:0Jforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourceDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourceEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$forward_simple_rnn_while_body_381752*0
cond(R&
$forward_simple_rnn_while_cond_381751*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
forward_simple_rnn/while?
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2E
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
5forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack!forward_simple_rnn/while:output:3Lforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype027
5forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
(forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(forward_simple_rnn/strided_slice_3/stack?
*forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*forward_simple_rnn/strided_slice_3/stack_1?
*forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_3/stack_2?
"forward_simple_rnn/strided_slice_3StridedSlice>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:01forward_simple_rnn/strided_slice_3/stack:output:03forward_simple_rnn/strided_slice_3/stack_1:output:03forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_3?
#forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#forward_simple_rnn/transpose_1/perm?
forward_simple_rnn/transpose_1	Transpose>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0,forward_simple_rnn/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2 
forward_simple_rnn/transpose_1n
backward_simple_rnn/ShapeShapeinputs_0*
T0*
_output_shapes
:2
backward_simple_rnn/Shape?
'backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_simple_rnn/strided_slice/stack?
)backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_1?
)backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_2?
!backward_simple_rnn/strided_sliceStridedSlice"backward_simple_rnn/Shape:output:00backward_simple_rnn/strided_slice/stack:output:02backward_simple_rnn/strided_slice/stack_1:output:02backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_simple_rnn/strided_slice?
backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2!
backward_simple_rnn/zeros/mul/y?
backward_simple_rnn/zeros/mulMul*backward_simple_rnn/strided_slice:output:0(backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/zeros/mul?
 backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 backward_simple_rnn/zeros/Less/y?
backward_simple_rnn/zeros/LessLess!backward_simple_rnn/zeros/mul:z:0)backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_simple_rnn/zeros/Less?
"backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2$
"backward_simple_rnn/zeros/packed/1?
 backward_simple_rnn/zeros/packedPack*backward_simple_rnn/strided_slice:output:0+backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_simple_rnn/zeros/packed?
backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
backward_simple_rnn/zeros/Const?
backward_simple_rnn/zerosFill)backward_simple_rnn/zeros/packed:output:0(backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
backward_simple_rnn/zeros?
"backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_simple_rnn/transpose/perm?
backward_simple_rnn/transpose	Transposeinputs_0+backward_simple_rnn/transpose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
backward_simple_rnn/transpose?
backward_simple_rnn/Shape_1Shape!backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
backward_simple_rnn/Shape_1?
)backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_1/stack?
+backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_1?
+backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_2?
#backward_simple_rnn/strided_slice_1StridedSlice$backward_simple_rnn/Shape_1:output:02backward_simple_rnn/strided_slice_1/stack:output:04backward_simple_rnn/strided_slice_1/stack_1:output:04backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_1?
/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/backward_simple_rnn/TensorArrayV2/element_shape?
!backward_simple_rnn/TensorArrayV2TensorListReserve8backward_simple_rnn/TensorArrayV2/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_simple_rnn/TensorArrayV2?
"backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2$
"backward_simple_rnn/ReverseV2/axis?
backward_simple_rnn/ReverseV2	ReverseV2!backward_simple_rnn/transpose:y:0+backward_simple_rnn/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
backward_simple_rnn/ReverseV2?
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????2K
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor&backward_simple_rnn/ReverseV2:output:0Rbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
)backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_2/stack?
+backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_1?
+backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_2?
#backward_simple_rnn/strided_slice_2StridedSlice!backward_simple_rnn/transpose:y:02backward_simple_rnn/strided_slice_2/stack:output:04backward_simple_rnn/strided_slice_2/stack_1:output:04backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_2?
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpDbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02=
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
,backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul,backward_simple_rnn/strided_slice_2:output:0Cbackward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,backward_simple_rnn/simple_rnn_cell_2/MatMul?
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02>
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
-backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd6backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0Dbackward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMul"backward_simple_rnn/zeros:output:0Ebackward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
20
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
)backward_simple_rnn/simple_rnn_cell_2/addAddV26backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:08backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2+
)backward_simple_rnn/simple_rnn_cell_2/add?
*backward_simple_rnn/simple_rnn_cell_2/TanhTanh-backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2,
*backward_simple_rnn/simple_rnn_cell_2/Tanh?
1backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   23
1backward_simple_rnn/TensorArrayV2_1/element_shape?
#backward_simple_rnn/TensorArrayV2_1TensorListReserve:backward_simple_rnn/TensorArrayV2_1/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#backward_simple_rnn/TensorArrayV2_1v
backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_simple_rnn/time?
,backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,backward_simple_rnn/while/maximum_iterations?
&backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_simple_rnn/while/loop_counter?
backward_simple_rnn/whileWhile/backward_simple_rnn/while/loop_counter:output:05backward_simple_rnn/while/maximum_iterations:output:0!backward_simple_rnn/time:output:0,backward_simple_rnn/TensorArrayV2_1:handle:0"backward_simple_rnn/zeros:output:0,backward_simple_rnn/strided_slice_1:output:0Kbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%backward_simple_rnn_while_body_381862*1
cond)R'
%backward_simple_rnn_while_cond_381861*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
backward_simple_rnn/while?
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2F
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
6backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack"backward_simple_rnn/while:output:3Mbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype028
6backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
)backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)backward_simple_rnn/strided_slice_3/stack?
+backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+backward_simple_rnn/strided_slice_3/stack_1?
+backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_3/stack_2?
#backward_simple_rnn/strided_slice_3StridedSlice?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:02backward_simple_rnn/strided_slice_3/stack:output:04backward_simple_rnn/strided_slice_3/stack_1:output:04backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_3?
$backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$backward_simple_rnn/transpose_1/perm?
backward_simple_rnn/transpose_1	Transpose?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0-backward_simple_rnn/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2!
backward_simple_rnn/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2+forward_simple_rnn/strided_slice_3:output:0,backward_simple_rnn/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp=^backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<^backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp>^backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp^backward_simple_rnn/while<^forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp=^forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp^forward_simple_rnn/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2|
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2z
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2~
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp26
backward_simple_rnn/whilebackward_simple_rnn/while2z
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp24
forward_simple_rnn/whileforward_simple_rnn/while:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
?
?
(__inference_dense_2_layer_call_fn_382611

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3803442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
W2.3-RNN-Reg-Bidirect-Swivel-Balance_bidirectional_backward_simple_rnn_while_cond_378245?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_loop_counter?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_maximum_iterationsW
Srnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholderY
Urnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholder_1Y
Urnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholder_2?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_less_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_strided_slice_1?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_cond_378245___redundant_placeholder0?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_cond_378245___redundant_placeholder1?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_cond_378245___redundant_placeholder2?
?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_cond_378245___redundant_placeholder3T
Prnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity
?
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/LessLessSrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_placeholder?rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_less_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2R
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Less?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/IdentityIdentityT2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity"?
Prnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_while_identity]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?F
?
$forward_simple_rnn_while_body_382200B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2A
=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0}
yforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
Z
Lforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
_
Mforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

%
!forward_simple_rnn_while_identity'
#forward_simple_rnn_while_identity_1'
#forward_simple_rnn_while_identity_2'
#forward_simple_rnn_while_identity_3'
#forward_simple_rnn_while_identity_4?
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1{
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor[
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
X
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0$forward_simple_rnn_while_placeholderSforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02>
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02B
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
1forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulCforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Hforward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
23
1forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02C
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAdd;forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0Iforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02D
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMul&forward_simple_rnn_while_placeholder_2Jforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
.forward_simple_rnn/while/simple_rnn_cell_1/addAddV2;forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0=forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
20
.forward_simple_rnn/while/simple_rnn_cell_1/add?
/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanh2forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
21
/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&forward_simple_rnn_while_placeholder_1$forward_simple_rnn_while_placeholder3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_simple_rnn/while/add/y?
forward_simple_rnn/while/addAddV2$forward_simple_rnn_while_placeholder'forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/while/add?
 forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 forward_simple_rnn/while/add_1/y?
forward_simple_rnn/while/add_1AddV2>forward_simple_rnn_while_forward_simple_rnn_while_loop_counter)forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
forward_simple_rnn/while/add_1?
!forward_simple_rnn/while/IdentityIdentity"forward_simple_rnn/while/add_1:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity?
#forward_simple_rnn/while/Identity_1IdentityDforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_1?
#forward_simple_rnn/while/Identity_2Identity forward_simple_rnn/while/add:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_2?
#forward_simple_rnn/while/Identity_3IdentityMforward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_3?
#forward_simple_rnn/while/Identity_4Identity3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0^forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2%
#forward_simple_rnn/while/Identity_4?
forward_simple_rnn/while/NoOpNoOpB^forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpA^forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpC^forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_simple_rnn/while/NoOp"|
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0"S
#forward_simple_rnn_while_identity_1,forward_simple_rnn/while/Identity_1:output:0"S
#forward_simple_rnn_while_identity_2,forward_simple_rnn/while/Identity_2:output:0"S
#forward_simple_rnn_while_identity_3,forward_simple_rnn/while/Identity_3:output:0"S
#forward_simple_rnn_while_identity_4,forward_simple_rnn/while/Identity_4:output:0"?
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourceLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensoryforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpAforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpBforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_379983

inputs+
forward_simple_rnn_379966:
'
forward_simple_rnn_379968:
+
forward_simple_rnn_379970:

,
backward_simple_rnn_379973:
(
backward_simple_rnn_379975:
,
backward_simple_rnn_379977:


identity??+backward_simple_rnn/StatefulPartitionedCall?*forward_simple_rnn/StatefulPartitionedCall?
*forward_simple_rnn/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_379966forward_simple_rnn_379968forward_simple_rnn_379970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_3799352,
*forward_simple_rnn/StatefulPartitionedCall?
+backward_simple_rnn/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_379973backward_simple_rnn_379975backward_simple_rnn_379977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_3798012-
+backward_simple_rnn/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV23forward_simple_rnn/StatefulPartitionedCall:output:04backward_simple_rnn/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp,^backward_simple_rnn/StatefulPartitionedCall+^forward_simple_rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2Z
+backward_simple_rnn/StatefulPartitionedCall+backward_simple_rnn/StatefulPartitionedCall2X
*forward_simple_rnn/StatefulPartitionedCall*forward_simple_rnn/StatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
while_cond_379868
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_379868___redundant_placeholder04
0while_while_cond_379868___redundant_placeholder14
0while_while_cond_379868___redundant_placeholder24
0while_while_cond_379868___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
C__inference_dense_2_layer_call_and_return_conditional_losses_380344

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?V
?
3bidirectional_backward_simple_rnn_while_body_381482`
\bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_loop_counterf
bbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_maximum_iterations7
3bidirectional_backward_simple_rnn_while_placeholder9
5bidirectional_backward_simple_rnn_while_placeholder_19
5bidirectional_backward_simple_rnn_while_placeholder_2_
[bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_strided_slice_1_0?
?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0l
Zbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
i
[bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
n
\bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

4
0bidirectional_backward_simple_rnn_while_identity6
2bidirectional_backward_simple_rnn_while_identity_16
2bidirectional_backward_simple_rnn_while_identity_26
2bidirectional_backward_simple_rnn_while_identity_36
2bidirectional_backward_simple_rnn_while_identity_4]
Ybidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_strided_slice_1?
?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensorj
Xbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
g
Ybidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
l
Zbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??Pbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Obidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Qbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Ybidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2[
Ybidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Kbidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_03bidirectional_backward_simple_rnn_while_placeholderbbidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02M
Kbidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Obidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpZbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02Q
Obidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
@bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulRbidirectional/backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Wbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2B
@bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
Pbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp[bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02R
Pbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
Abidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAddJbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Xbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2C
Abidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Qbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp\bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02S
Qbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Bbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul5bidirectional_backward_simple_rnn_while_placeholder_2Ybidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2D
Bbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
=bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2Jbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0Lbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2?
=bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/add?
>bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/TanhTanhAbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2@
>bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
Lbidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem5bidirectional_backward_simple_rnn_while_placeholder_13bidirectional_backward_simple_rnn_while_placeholderBbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02N
Lbidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
-bidirectional/backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-bidirectional/backward_simple_rnn/while/add/y?
+bidirectional/backward_simple_rnn/while/addAddV23bidirectional_backward_simple_rnn_while_placeholder6bidirectional/backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2-
+bidirectional/backward_simple_rnn/while/add?
/bidirectional/backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :21
/bidirectional/backward_simple_rnn/while/add_1/y?
-bidirectional/backward_simple_rnn/while/add_1AddV2\bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_loop_counter8bidirectional/backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2/
-bidirectional/backward_simple_rnn/while/add_1?
0bidirectional/backward_simple_rnn/while/IdentityIdentity1bidirectional/backward_simple_rnn/while/add_1:z:0-^bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 22
0bidirectional/backward_simple_rnn/while/Identity?
2bidirectional/backward_simple_rnn/while/Identity_1Identitybbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_maximum_iterations-^bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 24
2bidirectional/backward_simple_rnn/while/Identity_1?
2bidirectional/backward_simple_rnn/while/Identity_2Identity/bidirectional/backward_simple_rnn/while/add:z:0-^bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 24
2bidirectional/backward_simple_rnn/while/Identity_2?
2bidirectional/backward_simple_rnn/while/Identity_3Identity\bidirectional/backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^bidirectional/backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 24
2bidirectional/backward_simple_rnn/while/Identity_3?
2bidirectional/backward_simple_rnn/while/Identity_4IdentityBbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0-^bidirectional/backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
24
2bidirectional/backward_simple_rnn/while/Identity_4?
,bidirectional/backward_simple_rnn/while/NoOpNoOpQ^bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpP^bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpR^bidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2.
,bidirectional/backward_simple_rnn/while/NoOp"?
Ybidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_strided_slice_1[bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_strided_slice_1_0"m
0bidirectional_backward_simple_rnn_while_identity9bidirectional/backward_simple_rnn/while/Identity:output:0"q
2bidirectional_backward_simple_rnn_while_identity_1;bidirectional/backward_simple_rnn/while/Identity_1:output:0"q
2bidirectional_backward_simple_rnn_while_identity_2;bidirectional/backward_simple_rnn/while/Identity_2:output:0"q
2bidirectional_backward_simple_rnn_while_identity_3;bidirectional/backward_simple_rnn/while/Identity_3:output:0"q
2bidirectional_backward_simple_rnn_while_identity_4;bidirectional/backward_simple_rnn/while/Identity_4:output:0"?
Ybidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource[bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Zbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource\bidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Xbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceZbidirectional_backward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor?bidirectional_backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Pbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpPbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Obidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpObidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Qbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpQbidirectional/backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?G
?
%backward_simple_rnn_while_body_382086D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2C
?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0
{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
[
Mbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
`
Nbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

&
"backward_simple_rnn_while_identity(
$backward_simple_rnn_while_identity_1(
$backward_simple_rnn_while_identity_2(
$backward_simple_rnn_while_identity_3(
$backward_simple_rnn_while_identity_4A
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1}
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor\
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
Y
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????2M
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0%backward_simple_rnn_while_placeholderTbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02C
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
2backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulDbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Ibackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02D
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd<backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Jbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02E
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul'backward_simple_rnn_while_placeholder_2Kbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
26
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2<backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0>backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
21
/backward_simple_rnn/while/simple_rnn_cell_2/add?
0backward_simple_rnn/while/simple_rnn_cell_2/TanhTanh3backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
22
0backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'backward_simple_rnn_while_placeholder_1%backward_simple_rnn_while_placeholder4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02@
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_simple_rnn/while/add/y?
backward_simple_rnn/while/addAddV2%backward_simple_rnn_while_placeholder(backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/while/add?
!backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!backward_simple_rnn/while/add_1/y?
backward_simple_rnn/while/add_1AddV2@backward_simple_rnn_while_backward_simple_rnn_while_loop_counter*backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
backward_simple_rnn/while/add_1?
"backward_simple_rnn/while/IdentityIdentity#backward_simple_rnn/while/add_1:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity?
$backward_simple_rnn/while/Identity_1IdentityFbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_1?
$backward_simple_rnn/while/Identity_2Identity!backward_simple_rnn/while/add:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_2?
$backward_simple_rnn/while/Identity_3IdentityNbackward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_3?
$backward_simple_rnn/while/Identity_4Identity4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0^backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2&
$backward_simple_rnn/while/Identity_4?
backward_simple_rnn/while/NoOpNoOpC^backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpB^backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpD^backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
backward_simple_rnn/while/NoOp"?
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0"U
$backward_simple_rnn_while_identity_1-backward_simple_rnn/while/Identity_1:output:0"U
$backward_simple_rnn_while_identity_2-backward_simple_rnn/while/Identity_2:output:0"U
$backward_simple_rnn_while_identity_3-backward_simple_rnn/while/Identity_3:output:0"U
$backward_simple_rnn_while_identity_4-backward_simple_rnn/while/Identity_4:output:0"?
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpBbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpAbackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpCbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?I
?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_383053

inputsB
0simple_rnn_cell_1_matmul_readvariableop_resource:
?
1simple_rnn_cell_1_biasadd_readvariableop_resource:
D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_1/BiasAdd/ReadVariableOp?'simple_rnn_cell_1/MatMul/ReadVariableOp?)simple_rnn_cell_1/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOp?
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul?
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOp?
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/BiasAdd?
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOp?
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul_1?
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/add?
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_382987*
condR
while_cond_382986*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
while_cond_382762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_382762___redundant_placeholder04
0while_while_cond_382762___redundant_placeholder14
0while_while_cond_382762___redundant_placeholder24
0while_while_cond_382762___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
4__inference_backward_simple_rnn_layer_call_fn_383209

inputs
unknown:

	unknown_0:

	unknown_1:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_3798012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_381020

inputs
unknown
	unknown_0	
	unknown_1:
??
	unknown_2:

	unknown_3:

	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:
	unknown_9:

unknown_10:

unknown_11:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_3803792
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?{
?
V2.3-RNN-Reg-Bidirect-Swivel-Balance_bidirectional_forward_simple_rnn_while_body_378136?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_loop_counter?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_maximum_iterationsV
Rrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholderX
Trnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholder_1X
Trnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholder_2?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_strided_slice_1_0?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0?
yrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
?
zrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
?
{rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

S
Ornn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identityU
Qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity_1U
Qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity_2U
Qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity_3U
Qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity_4?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_strided_slice_1?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor?
wrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
?
xrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
?
yrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??s2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?r2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
|2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2~
|2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0Rrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholder?2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02p
n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
r2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpyrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02t
r2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulu2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2e
c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
s2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpzrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02u
s2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAddm2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0{2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2f
d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp{rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02v
t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMulTrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholder_2|2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2g
e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
`2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/addAddV2m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2b
`2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/add?
a2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanhd2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2c
a2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemTrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholder_1Rrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholdere2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02q
o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2R
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add/y?
N2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/addAddV2Rrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_placeholderY2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2P
N2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add?
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2T
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add_1/y?
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add_1AddV2?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_loop_counter[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2R
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add_1?
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/IdentityIdentityT2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add_1:z:0P^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2U
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_1Identity?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_maximum_iterationsP^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_1?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_2IdentityR2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/add:z:0P^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_2?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_3Identity2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0P^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_3?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_4Identitye2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0P^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_4?
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/NoOpNoOpt^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOps^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpu^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2Q
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/NoOp"?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_strided_slice_1?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_strided_slice_1_0"?
Ornn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity:output:0"?
Qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity_1^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_1:output:0"?
Qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity_2^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_2:output:0"?
Qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity_3^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_3:output:0"?
Qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_identity_4^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/Identity_4:output:0"?
xrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourcezrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
yrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource{rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
wrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceyrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor?rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_2_3_rnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
s2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOps2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
r2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpr2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpt2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
? 
?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_380914	
input
embed_380881
embed_380883	 
embed_380885:
??&
bidirectional_380889:
"
bidirectional_380891:
&
bidirectional_380893:

&
bidirectional_380895:
"
bidirectional_380897:
&
bidirectional_380899:

 
dense_2_380902:
dense_2_380904: 
dense_3_380908:
dense_3_380910:
identity??%bidirectional/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?embed/StatefulPartitionedCall?
embed/StatefulPartitionedCallStatefulPartitionedCallinputembed_380881embed_380883embed_380885*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_embed_layer_call_and_return_conditional_losses_3800732
embed/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall&embed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3800942
reshape/PartitionedCall?
%bidirectional/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0bidirectional_380889bidirectional_380891bidirectional_380893bidirectional_380895bidirectional_380897bidirectional_380899*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_bidirectional_layer_call_and_return_conditional_losses_3803192'
%bidirectional/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_2_380902dense_2_380904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3803442!
dense_2/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3803552
dropout_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_380908dense_3_380910*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3803722!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^bidirectional/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^embed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_nameInput:

_output_shapes
: 
?
?
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_383772

inputs
states_00
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????
2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityg

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_382649

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_380355

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%backward_simple_rnn_while_cond_380250D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2F
Bbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_380250___redundant_placeholder0\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_380250___redundant_placeholder1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_380250___redundant_placeholder2\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_380250___redundant_placeholder3&
"backward_simple_rnn_while_identity
?
backward_simple_rnn/while/LessLess%backward_simple_rnn_while_placeholderBbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2 
backward_simple_rnn/while/Less?
"backward_simple_rnn/while/IdentityIdentity"backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
V
)__inference_restored_function_body_383840
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *!
fR
__inference__creator_1282
StatefulPartitionedCallj
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
?
while_cond_378933
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_378933___redundant_placeholder04
0while_while_cond_378933___redundant_placeholder14
0while_while_cond_378933___redundant_placeholder24
0while_while_cond_378933___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?	
?
.__inference_bidirectional_layer_call_fn_381706

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_bidirectional_layer_call_and_return_conditional_losses_3806942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3bidirectional_backward_simple_rnn_while_cond_381222`
\bidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_loop_counterf
bbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_maximum_iterations7
3bidirectional_backward_simple_rnn_while_placeholder9
5bidirectional_backward_simple_rnn_while_placeholder_19
5bidirectional_backward_simple_rnn_while_placeholder_2b
^bidirectional_backward_simple_rnn_while_less_bidirectional_backward_simple_rnn_strided_slice_1x
tbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_cond_381222___redundant_placeholder0x
tbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_cond_381222___redundant_placeholder1x
tbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_cond_381222___redundant_placeholder2x
tbidirectional_backward_simple_rnn_while_bidirectional_backward_simple_rnn_while_cond_381222___redundant_placeholder34
0bidirectional_backward_simple_rnn_while_identity
?
,bidirectional/backward_simple_rnn/while/LessLess3bidirectional_backward_simple_rnn_while_placeholder^bidirectional_backward_simple_rnn_while_less_bidirectional_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2.
,bidirectional/backward_simple_rnn/while/Less?
0bidirectional/backward_simple_rnn/while/IdentityIdentity0bidirectional/backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 22
0bidirectional/backward_simple_rnn/while/Identity"m
0bidirectional_backward_simple_rnn_while_identity9bidirectional/backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
4__inference_backward_simple_rnn_layer_call_fn_383176
inputs_0
unknown:

	unknown_0:

	unknown_1:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_3789972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?U
?
2bidirectional_forward_simple_rnn_while_body_381113^
Zbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_loop_counterd
`bidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_maximum_iterations6
2bidirectional_forward_simple_rnn_while_placeholder8
4bidirectional_forward_simple_rnn_while_placeholder_18
4bidirectional_forward_simple_rnn_while_placeholder_2]
Ybidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_strided_slice_1_0?
?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0k
Ybidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
h
Zbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
m
[bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

3
/bidirectional_forward_simple_rnn_while_identity5
1bidirectional_forward_simple_rnn_while_identity_15
1bidirectional_forward_simple_rnn_while_identity_25
1bidirectional_forward_simple_rnn_while_identity_35
1bidirectional_forward_simple_rnn_while_identity_4[
Wbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_strided_slice_1?
?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensori
Wbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
f
Xbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
k
Ybidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??Obidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?Nbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?Pbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Xbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2Z
Xbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Jbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_02bidirectional_forward_simple_rnn_while_placeholderabidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02L
Jbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Nbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpYbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02P
Nbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
?bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulQbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Vbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2A
?bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
Obidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpZbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02Q
Obidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
@bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAddIbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0Wbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2B
@bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
Pbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp[bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02R
Pbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Abidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMul4bidirectional_forward_simple_rnn_while_placeholder_2Xbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2C
Abidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
<bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/addAddV2Ibidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0Kbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2>
<bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/add?
=bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanh@bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2?
=bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
Kbidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem4bidirectional_forward_simple_rnn_while_placeholder_12bidirectional_forward_simple_rnn_while_placeholderAbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02M
Kbidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
,bidirectional/forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,bidirectional/forward_simple_rnn/while/add/y?
*bidirectional/forward_simple_rnn/while/addAddV22bidirectional_forward_simple_rnn_while_placeholder5bidirectional/forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2,
*bidirectional/forward_simple_rnn/while/add?
.bidirectional/forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.bidirectional/forward_simple_rnn/while/add_1/y?
,bidirectional/forward_simple_rnn/while/add_1AddV2Zbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_loop_counter7bidirectional/forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2.
,bidirectional/forward_simple_rnn/while/add_1?
/bidirectional/forward_simple_rnn/while/IdentityIdentity0bidirectional/forward_simple_rnn/while/add_1:z:0,^bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 21
/bidirectional/forward_simple_rnn/while/Identity?
1bidirectional/forward_simple_rnn/while/Identity_1Identity`bidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_maximum_iterations,^bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 23
1bidirectional/forward_simple_rnn/while/Identity_1?
1bidirectional/forward_simple_rnn/while/Identity_2Identity.bidirectional/forward_simple_rnn/while/add:z:0,^bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 23
1bidirectional/forward_simple_rnn/while/Identity_2?
1bidirectional/forward_simple_rnn/while/Identity_3Identity[bidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 23
1bidirectional/forward_simple_rnn/while/Identity_3?
1bidirectional/forward_simple_rnn/while/Identity_4IdentityAbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0,^bidirectional/forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
23
1bidirectional/forward_simple_rnn/while/Identity_4?
+bidirectional/forward_simple_rnn/while/NoOpNoOpP^bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpO^bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpQ^bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2-
+bidirectional/forward_simple_rnn/while/NoOp"?
Wbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_strided_slice_1Ybidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_strided_slice_1_0"k
/bidirectional_forward_simple_rnn_while_identity8bidirectional/forward_simple_rnn/while/Identity:output:0"o
1bidirectional_forward_simple_rnn_while_identity_1:bidirectional/forward_simple_rnn/while/Identity_1:output:0"o
1bidirectional_forward_simple_rnn_while_identity_2:bidirectional/forward_simple_rnn/while/Identity_2:output:0"o
1bidirectional_forward_simple_rnn_while_identity_3:bidirectional/forward_simple_rnn/while/Identity_3:output:0"o
1bidirectional_forward_simple_rnn_while_identity_4:bidirectional/forward_simple_rnn/while/Identity_4:output:0"?
Xbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourceZbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
Ybidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource[bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
Wbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceYbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Obidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpObidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
Nbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpNbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Pbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpPbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_forward_simple_rnn_layer_call_fn_382684
inputs_0
unknown:

	unknown_0:

	unknown_1:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_3784612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?G
?
%backward_simple_rnn_while_body_382534D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2C
?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0
{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
[
Mbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
`
Nbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

&
"backward_simple_rnn_while_identity(
$backward_simple_rnn_while_identity_1(
$backward_simple_rnn_while_identity_2(
$backward_simple_rnn_while_identity_3(
$backward_simple_rnn_while_identity_4A
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1}
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor\
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
Y
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0%backward_simple_rnn_while_placeholderTbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02C
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
2backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulDbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Ibackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02D
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd<backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Jbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02E
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul'backward_simple_rnn_while_placeholder_2Kbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
26
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2<backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0>backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
21
/backward_simple_rnn/while/simple_rnn_cell_2/add?
0backward_simple_rnn/while/simple_rnn_cell_2/TanhTanh3backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
22
0backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'backward_simple_rnn_while_placeholder_1%backward_simple_rnn_while_placeholder4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02@
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_simple_rnn/while/add/y?
backward_simple_rnn/while/addAddV2%backward_simple_rnn_while_placeholder(backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/while/add?
!backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!backward_simple_rnn/while/add_1/y?
backward_simple_rnn/while/add_1AddV2@backward_simple_rnn_while_backward_simple_rnn_while_loop_counter*backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
backward_simple_rnn/while/add_1?
"backward_simple_rnn/while/IdentityIdentity#backward_simple_rnn/while/add_1:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity?
$backward_simple_rnn/while/Identity_1IdentityFbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_1?
$backward_simple_rnn/while/Identity_2Identity!backward_simple_rnn/while/add:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_2?
$backward_simple_rnn/while/Identity_3IdentityNbackward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_3?
$backward_simple_rnn/while/Identity_4Identity4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0^backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2&
$backward_simple_rnn/while/Identity_4?
backward_simple_rnn/while/NoOpNoOpC^backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpB^backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpD^backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
backward_simple_rnn/while/NoOp"?
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0"U
$backward_simple_rnn_while_identity_1-backward_simple_rnn/while/Identity_1:output:0"U
$backward_simple_rnn_while_identity_2-backward_simple_rnn/while/Identity_2:output:0"U
$backward_simple_rnn_while_identity_3-backward_simple_rnn/while/Identity_3:output:0"U
$backward_simple_rnn_while_identity_4-backward_simple_rnn/while/Identity_4:output:0"?
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpBbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpAbackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpCbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
%backward_simple_rnn_while_cond_380625D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2F
Bbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_380625___redundant_placeholder0\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_380625___redundant_placeholder1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_380625___redundant_placeholder2\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_380625___redundant_placeholder3&
"backward_simple_rnn_while_identity
?
backward_simple_rnn/while/LessLess%backward_simple_rnn_while_placeholderBbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2 
backward_simple_rnn/while/Less?
"backward_simple_rnn/while/IdentityIdentity"backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_382637

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
while_body_382987
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_1/MatMul/ReadVariableOp?/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOp?
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_1/MatMul?
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_1/BiasAdd?
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_1/MatMul_1?
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/add?
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
$forward_simple_rnn_while_cond_380140B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2D
@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_380140___redundant_placeholder0Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_380140___redundant_placeholder1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_380140___redundant_placeholder2Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_380140___redundant_placeholder3%
!forward_simple_rnn_while_identity
?
forward_simple_rnn/while/LessLess$forward_simple_rnn_while_placeholder@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
forward_simple_rnn/while/Less?
!forward_simple_rnn/while/IdentityIdentity!forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
2bidirectional_forward_simple_rnn_while_cond_381112^
Zbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_loop_counterd
`bidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_maximum_iterations6
2bidirectional_forward_simple_rnn_while_placeholder8
4bidirectional_forward_simple_rnn_while_placeholder_18
4bidirectional_forward_simple_rnn_while_placeholder_2`
\bidirectional_forward_simple_rnn_while_less_bidirectional_forward_simple_rnn_strided_slice_1v
rbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_cond_381112___redundant_placeholder0v
rbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_cond_381112___redundant_placeholder1v
rbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_cond_381112___redundant_placeholder2v
rbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_cond_381112___redundant_placeholder33
/bidirectional_forward_simple_rnn_while_identity
?
+bidirectional/forward_simple_rnn/while/LessLess2bidirectional_forward_simple_rnn_while_placeholder\bidirectional_forward_simple_rnn_while_less_bidirectional_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2-
+bidirectional/forward_simple_rnn/while/Less?
/bidirectional/forward_simple_rnn/while/IdentityIdentity/bidirectional/forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 21
/bidirectional/forward_simple_rnn/while/Identity"k
/bidirectional_forward_simple_rnn_while_identity8bidirectional/forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
? 
?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_380379

inputs
embed_380074
embed_380076	 
embed_380078:
??&
bidirectional_380320:
"
bidirectional_380322:
&
bidirectional_380324:

&
bidirectional_380326:
"
bidirectional_380328:
&
bidirectional_380330:

 
dense_2_380345:
dense_2_380347: 
dense_3_380373:
dense_3_380375:
identity??%bidirectional/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?embed/StatefulPartitionedCall?
embed/StatefulPartitionedCallStatefulPartitionedCallinputsembed_380074embed_380076embed_380078*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_embed_layer_call_and_return_conditional_losses_3800732
embed/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall&embed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3800942
reshape/PartitionedCall?
%bidirectional/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0bidirectional_380320bidirectional_380322bidirectional_380324bidirectional_380326bidirectional_380328bidirectional_380330*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_bidirectional_layer_call_and_return_conditional_losses_3803192'
%bidirectional/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_2_380345dense_2_380347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3803442!
dense_2/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3803552
dropout_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_380373dense_3_380375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3803722!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^bidirectional/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^embed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
A__inference_embed_layer_call_and_return_conditional_losses_381620

inputs
unknown
	unknown_0	
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_2217052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_379659

inputs+
forward_simple_rnn_379528:
'
forward_simple_rnn_379530:
+
forward_simple_rnn_379532:

,
backward_simple_rnn_379649:
(
backward_simple_rnn_379651:
,
backward_simple_rnn_379653:


identity??+backward_simple_rnn/StatefulPartitionedCall?*forward_simple_rnn/StatefulPartitionedCall?
*forward_simple_rnn/StatefulPartitionedCallStatefulPartitionedCallinputsforward_simple_rnn_379528forward_simple_rnn_379530forward_simple_rnn_379532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_3795272,
*forward_simple_rnn/StatefulPartitionedCall?
+backward_simple_rnn/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_simple_rnn_379649backward_simple_rnn_379651backward_simple_rnn_379653*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_3796482-
+backward_simple_rnn/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV23forward_simple_rnn/StatefulPartitionedCall:output:04backward_simple_rnn/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp,^backward_simple_rnn/StatefulPartitionedCall+^forward_simple_rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 2Z
+backward_simple_rnn/StatefulPartitionedCall+backward_simple_rnn/StatefulPartitionedCall2X
*forward_simple_rnn/StatefulPartitionedCall*forward_simple_rnn/StatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
%backward_simple_rnn_while_cond_381861D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2F
Bbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_381861___redundant_placeholder0\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_381861___redundant_placeholder1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_381861___redundant_placeholder2\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_381861___redundant_placeholder3&
"backward_simple_rnn_while_identity
?
backward_simple_rnn/while/LessLess%backward_simple_rnn_while_placeholderBbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2 
backward_simple_rnn/while/Less?
"backward_simple_rnn/while/IdentityIdentity"backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?I
?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_382941
inputs_0B
0simple_rnn_cell_1_matmul_readvariableop_resource:
?
1simple_rnn_cell_1_biasadd_readvariableop_resource:
D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_1/BiasAdd/ReadVariableOp?'simple_rnn_cell_1/MatMul/ReadVariableOp?)simple_rnn_cell_1/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOp?
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul?
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOp?
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/BiasAdd?
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOp?
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul_1?
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/add?
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_382875*
condR
while_cond_382874*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
2__inference_simple_rnn_cell_1_layer_call_fn_383693

inputs
states_0
unknown:

	unknown_0:

	unknown_1:


identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_3785052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0
?
?
3__inference_forward_simple_rnn_layer_call_fn_382695
inputs_0
unknown:

	unknown_0:

	unknown_1:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_3786352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
C__inference_dense_3_layer_call_and_return_conditional_losses_382673

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
ReluS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1?
clip_by_value/MinimumMinimumRelu:activations:0Const:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimum?
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuel
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
while_body_383371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?1
?
while_body_383099
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_1_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_1_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_1/MatMul/ReadVariableOp?/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_1/MatMul/ReadVariableOp?
while/simple_rnn_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_1/MatMul?
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_1/BiasAddBiasAdd(while/simple_rnn_cell_1/MatMul:product:06while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_1/BiasAdd?
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_1/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_1/MatMul_1?
while/simple_rnn_cell_1/addAddV2(while/simple_rnn_cell_1/BiasAdd:output:0*while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/add?
while/simple_rnn_cell_1/TanhTanhwhile/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_1/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_1/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_1/MatMul/ReadVariableOp0^while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_1_biasadd_readvariableop_resource9while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_1_matmul_1_readvariableop_resource:while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_1_matmul_readvariableop_resource8while_simple_rnn_cell_1_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp.while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_1/MatMul/ReadVariableOp-while/simple_rnn_cell_1/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
??
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_380319

inputsU
Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
R
Dforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
W
Eforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

V
Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
S
Ebackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
X
Fbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?backward_simple_rnn/while?;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?forward_simple_rnn/whilej
forward_simple_rnn/ShapeShapeinputs*
T0*
_output_shapes
:2
forward_simple_rnn/Shape?
&forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_simple_rnn/strided_slice/stack?
(forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_1?
(forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_2?
 forward_simple_rnn/strided_sliceStridedSlice!forward_simple_rnn/Shape:output:0/forward_simple_rnn/strided_slice/stack:output:01forward_simple_rnn/strided_slice/stack_1:output:01forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_simple_rnn/strided_slice?
forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2 
forward_simple_rnn/zeros/mul/y?
forward_simple_rnn/zeros/mulMul)forward_simple_rnn/strided_slice:output:0'forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/mul?
forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
forward_simple_rnn/zeros/Less/y?
forward_simple_rnn/zeros/LessLess forward_simple_rnn/zeros/mul:z:0(forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/Less?
!forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!forward_simple_rnn/zeros/packed/1?
forward_simple_rnn/zeros/packedPack)forward_simple_rnn/strided_slice:output:0*forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_simple_rnn/zeros/packed?
forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
forward_simple_rnn/zeros/Const?
forward_simple_rnn/zerosFill(forward_simple_rnn/zeros/packed:output:0'forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
forward_simple_rnn/zeros?
!forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_simple_rnn/transpose/perm?
forward_simple_rnn/transpose	Transposeinputs*forward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
forward_simple_rnn/transpose?
forward_simple_rnn/Shape_1Shape forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
forward_simple_rnn/Shape_1?
(forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_1/stack?
*forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_1?
*forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_2?
"forward_simple_rnn/strided_slice_1StridedSlice#forward_simple_rnn/Shape_1:output:01forward_simple_rnn/strided_slice_1/stack:output:03forward_simple_rnn/strided_slice_1/stack_1:output:03forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_1?
.forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.forward_simple_rnn/TensorArrayV2/element_shape?
 forward_simple_rnn/TensorArrayV2TensorListReserve7forward_simple_rnn/TensorArrayV2/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_simple_rnn/TensorArrayV2?
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2J
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor forward_simple_rnn/transpose:y:0Qforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
(forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_2/stack?
*forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_1?
*forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_2?
"forward_simple_rnn/strided_slice_2StridedSlice forward_simple_rnn/transpose:y:01forward_simple_rnn/strided_slice_2/stack:output:03forward_simple_rnn/strided_slice_2/stack_1:output:03forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_2?
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpCforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02<
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
+forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul+forward_simple_rnn/strided_slice_2:output:0Bforward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2-
+forward_simple_rnn/simple_rnn_cell_1/MatMul?
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02=
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
,forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAdd5forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0Cforward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02>
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMul!forward_simple_rnn/zeros:output:0Dforward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
(forward_simple_rnn/simple_rnn_cell_1/addAddV25forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:07forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2*
(forward_simple_rnn/simple_rnn_cell_1/add?
)forward_simple_rnn/simple_rnn_cell_1/TanhTanh,forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2+
)forward_simple_rnn/simple_rnn_cell_1/Tanh?
0forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0forward_simple_rnn/TensorArrayV2_1/element_shape?
"forward_simple_rnn/TensorArrayV2_1TensorListReserve9forward_simple_rnn/TensorArrayV2_1/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"forward_simple_rnn/TensorArrayV2_1t
forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_simple_rnn/time?
+forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+forward_simple_rnn/while/maximum_iterations?
%forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%forward_simple_rnn/while/loop_counter?
forward_simple_rnn/whileWhile.forward_simple_rnn/while/loop_counter:output:04forward_simple_rnn/while/maximum_iterations:output:0 forward_simple_rnn/time:output:0+forward_simple_rnn/TensorArrayV2_1:handle:0!forward_simple_rnn/zeros:output:0+forward_simple_rnn/strided_slice_1:output:0Jforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourceDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourceEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$forward_simple_rnn_while_body_380141*0
cond(R&
$forward_simple_rnn_while_cond_380140*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
forward_simple_rnn/while?
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2E
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
5forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack!forward_simple_rnn/while:output:3Lforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype027
5forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
(forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(forward_simple_rnn/strided_slice_3/stack?
*forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*forward_simple_rnn/strided_slice_3/stack_1?
*forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_3/stack_2?
"forward_simple_rnn/strided_slice_3StridedSlice>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:01forward_simple_rnn/strided_slice_3/stack:output:03forward_simple_rnn/strided_slice_3/stack_1:output:03forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_3?
#forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#forward_simple_rnn/transpose_1/perm?
forward_simple_rnn/transpose_1	Transpose>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0,forward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2 
forward_simple_rnn/transpose_1l
backward_simple_rnn/ShapeShapeinputs*
T0*
_output_shapes
:2
backward_simple_rnn/Shape?
'backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_simple_rnn/strided_slice/stack?
)backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_1?
)backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_2?
!backward_simple_rnn/strided_sliceStridedSlice"backward_simple_rnn/Shape:output:00backward_simple_rnn/strided_slice/stack:output:02backward_simple_rnn/strided_slice/stack_1:output:02backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_simple_rnn/strided_slice?
backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2!
backward_simple_rnn/zeros/mul/y?
backward_simple_rnn/zeros/mulMul*backward_simple_rnn/strided_slice:output:0(backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/zeros/mul?
 backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 backward_simple_rnn/zeros/Less/y?
backward_simple_rnn/zeros/LessLess!backward_simple_rnn/zeros/mul:z:0)backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_simple_rnn/zeros/Less?
"backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2$
"backward_simple_rnn/zeros/packed/1?
 backward_simple_rnn/zeros/packedPack*backward_simple_rnn/strided_slice:output:0+backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_simple_rnn/zeros/packed?
backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
backward_simple_rnn/zeros/Const?
backward_simple_rnn/zerosFill)backward_simple_rnn/zeros/packed:output:0(backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
backward_simple_rnn/zeros?
"backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_simple_rnn/transpose/perm?
backward_simple_rnn/transpose	Transposeinputs+backward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
backward_simple_rnn/transpose?
backward_simple_rnn/Shape_1Shape!backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
backward_simple_rnn/Shape_1?
)backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_1/stack?
+backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_1?
+backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_2?
#backward_simple_rnn/strided_slice_1StridedSlice$backward_simple_rnn/Shape_1:output:02backward_simple_rnn/strided_slice_1/stack:output:04backward_simple_rnn/strided_slice_1/stack_1:output:04backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_1?
/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/backward_simple_rnn/TensorArrayV2/element_shape?
!backward_simple_rnn/TensorArrayV2TensorListReserve8backward_simple_rnn/TensorArrayV2/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_simple_rnn/TensorArrayV2?
"backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2$
"backward_simple_rnn/ReverseV2/axis?
backward_simple_rnn/ReverseV2	ReverseV2!backward_simple_rnn/transpose:y:0+backward_simple_rnn/ReverseV2/axis:output:0*
T0*+
_output_shapes
:?????????2
backward_simple_rnn/ReverseV2?
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2K
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor&backward_simple_rnn/ReverseV2:output:0Rbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
)backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_2/stack?
+backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_1?
+backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_2?
#backward_simple_rnn/strided_slice_2StridedSlice!backward_simple_rnn/transpose:y:02backward_simple_rnn/strided_slice_2/stack:output:04backward_simple_rnn/strided_slice_2/stack_1:output:04backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_2?
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpDbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02=
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
,backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul,backward_simple_rnn/strided_slice_2:output:0Cbackward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,backward_simple_rnn/simple_rnn_cell_2/MatMul?
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02>
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
-backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd6backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0Dbackward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMul"backward_simple_rnn/zeros:output:0Ebackward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
20
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
)backward_simple_rnn/simple_rnn_cell_2/addAddV26backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:08backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2+
)backward_simple_rnn/simple_rnn_cell_2/add?
*backward_simple_rnn/simple_rnn_cell_2/TanhTanh-backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2,
*backward_simple_rnn/simple_rnn_cell_2/Tanh?
1backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   23
1backward_simple_rnn/TensorArrayV2_1/element_shape?
#backward_simple_rnn/TensorArrayV2_1TensorListReserve:backward_simple_rnn/TensorArrayV2_1/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#backward_simple_rnn/TensorArrayV2_1v
backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_simple_rnn/time?
,backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,backward_simple_rnn/while/maximum_iterations?
&backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_simple_rnn/while/loop_counter?
backward_simple_rnn/whileWhile/backward_simple_rnn/while/loop_counter:output:05backward_simple_rnn/while/maximum_iterations:output:0!backward_simple_rnn/time:output:0,backward_simple_rnn/TensorArrayV2_1:handle:0"backward_simple_rnn/zeros:output:0,backward_simple_rnn/strided_slice_1:output:0Kbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%backward_simple_rnn_while_body_380251*1
cond)R'
%backward_simple_rnn_while_cond_380250*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
backward_simple_rnn/while?
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2F
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
6backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack"backward_simple_rnn/while:output:3Mbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype028
6backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
)backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)backward_simple_rnn/strided_slice_3/stack?
+backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+backward_simple_rnn/strided_slice_3/stack_1?
+backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_3/stack_2?
#backward_simple_rnn/strided_slice_3StridedSlice?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:02backward_simple_rnn/strided_slice_3/stack:output:04backward_simple_rnn/strided_slice_3/stack_1:output:04backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_3?
$backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$backward_simple_rnn/transpose_1/perm?
backward_simple_rnn/transpose_1	Transpose?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0-backward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2!
backward_simple_rnn/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2+forward_simple_rnn/strided_slice_3:output:0,backward_simple_rnn/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp=^backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<^backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp>^backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp^backward_simple_rnn/while<^forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp=^forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp^forward_simple_rnn/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2|
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2z
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2~
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp26
backward_simple_rnn/whilebackward_simple_rnn/while2z
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp24
forward_simple_rnn/whileforward_simple_rnn/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?@
?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_379173

inputs*
simple_rnn_cell_2_379098:
&
simple_rnn_cell_2_379100:
*
simple_rnn_cell_2_379102:


identity??)simple_rnn_cell_2/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????2
	ReverseV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
)simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_2_379098simple_rnn_cell_2_379100simple_rnn_cell_2_379102*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_3790412+
)simple_rnn_cell_2/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_2_379098simple_rnn_cell_2_379100simple_rnn_cell_2_379102*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_379110*
condR
while_cond_379109*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp*^simple_rnn_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2V
)simple_rnn_cell_2/StatefulPartitionedCall)simple_rnn_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
.__inference_bidirectional_layer_call_fn_381655
inputs_0
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_bidirectional_layer_call_and_return_conditional_losses_3796592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
?
?
%backward_simple_rnn_while_cond_382309D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2F
Bbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382309___redundant_placeholder0\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382309___redundant_placeholder1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382309___redundant_placeholder2\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382309___redundant_placeholder3&
"backward_simple_rnn_while_identity
?
backward_simple_rnn/while/LessLess%backward_simple_rnn_while_placeholderBbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2 
backward_simple_rnn/while/Less?
"backward_simple_rnn/while/IdentityIdentity"backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
$forward_simple_rnn_while_cond_380515B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2D
@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_380515___redundant_placeholder0Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_380515___redundant_placeholder1Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_380515___redundant_placeholder2Z
Vforward_simple_rnn_while_forward_simple_rnn_while_cond_380515___redundant_placeholder3%
!forward_simple_rnn_while_identity
?
forward_simple_rnn/while/LessLess$forward_simple_rnn_while_placeholder@forward_simple_rnn_while_less_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2
forward_simple_rnn/while/Less?
!forward_simple_rnn/while/IdentityIdentity!forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_379460
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_379460___redundant_placeholder04
0while_while_cond_379460___redundant_placeholder14
0while_while_cond_379460___redundant_placeholder24
0while_while_cond_379460___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?1
?
while_body_379582
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_dense_3_layer_call_fn_382658

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3803722
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_restored_function_body_221705
	sentences
unknown
	unknown_0	
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	sentencesunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference___call___1152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:?????????
#
_user_specified_name	sentences:

_output_shapes
: 
?
F
*__inference_dropout_1_layer_call_fn_382627

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3803552
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
2__inference_simple_rnn_cell_2_layer_call_fn_383755

inputs
states_0
unknown:

	unknown_0:

	unknown_1:


identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_3790412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0
?!
?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_380950	
input
embed_380917
embed_380919	 
embed_380921:
??&
bidirectional_380925:
"
bidirectional_380927:
&
bidirectional_380929:

&
bidirectional_380931:
"
bidirectional_380933:
&
bidirectional_380935:

 
dense_2_380938:
dense_2_380940: 
dense_3_380944:
dense_3_380946:
identity??%bidirectional/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?embed/StatefulPartitionedCall?
embed/StatefulPartitionedCallStatefulPartitionedCallinputembed_380917embed_380919embed_380921*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_embed_layer_call_and_return_conditional_losses_3807402
embed/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall&embed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3800942
reshape/PartitionedCall?
%bidirectional/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0bidirectional_380925bidirectional_380927bidirectional_380929bidirectional_380931bidirectional_380933bidirectional_380935*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_bidirectional_layer_call_and_return_conditional_losses_3806942'
%bidirectional/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_2_380938dense_2_380940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3803442!
dense_2/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3804382#
!dropout_1/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_380944dense_3_380946*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3803722!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^bidirectional/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^embed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_nameInput:

_output_shapes
: 
?
?
while_cond_383484
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_383484___redundant_placeholder04
0while_while_cond_383484___redundant_placeholder14
0while_while_cond_383484___redundant_placeholder24
0while_while_cond_383484___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?1
?
while_body_383599
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_383598
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_383598___redundant_placeholder04
0while_while_cond_383598___redundant_placeholder14
0while_while_cond_383598___redundant_placeholder24
0while_while_cond_383598___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?F
?
$forward_simple_rnn_while_body_381976B
>forward_simple_rnn_while_forward_simple_rnn_while_loop_counterH
Dforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations(
$forward_simple_rnn_while_placeholder*
&forward_simple_rnn_while_placeholder_1*
&forward_simple_rnn_while_placeholder_2A
=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0}
yforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
Z
Lforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
_
Mforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

%
!forward_simple_rnn_while_identity'
#forward_simple_rnn_while_identity_1'
#forward_simple_rnn_while_identity_2'
#forward_simple_rnn_while_identity_3'
#forward_simple_rnn_while_identity_4?
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1{
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor[
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
X
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
]
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????2L
Jforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0$forward_simple_rnn_while_placeholderSforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02>
<forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02B
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
1forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulCforward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Hforward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
23
1forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02C
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAdd;forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0Iforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02D
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMul&forward_simple_rnn_while_placeholder_2Jforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
.forward_simple_rnn/while/simple_rnn_cell_1/addAddV2;forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0=forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
20
.forward_simple_rnn/while/simple_rnn_cell_1/add?
/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanh2forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
21
/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&forward_simple_rnn_while_placeholder_1$forward_simple_rnn_while_placeholder3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02?
=forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
forward_simple_rnn/while/add/y?
forward_simple_rnn/while/addAddV2$forward_simple_rnn_while_placeholder'forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/while/add?
 forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 forward_simple_rnn/while/add_1/y?
forward_simple_rnn/while/add_1AddV2>forward_simple_rnn_while_forward_simple_rnn_while_loop_counter)forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
forward_simple_rnn/while/add_1?
!forward_simple_rnn/while/IdentityIdentity"forward_simple_rnn/while/add_1:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2#
!forward_simple_rnn/while/Identity?
#forward_simple_rnn/while/Identity_1IdentityDforward_simple_rnn_while_forward_simple_rnn_while_maximum_iterations^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_1?
#forward_simple_rnn/while/Identity_2Identity forward_simple_rnn/while/add:z:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_2?
#forward_simple_rnn/while/Identity_3IdentityMforward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2%
#forward_simple_rnn/while/Identity_3?
#forward_simple_rnn/while/Identity_4Identity3forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0^forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2%
#forward_simple_rnn/while/Identity_4?
forward_simple_rnn/while/NoOpNoOpB^forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpA^forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpC^forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
forward_simple_rnn/while/NoOp"|
;forward_simple_rnn_while_forward_simple_rnn_strided_slice_1=forward_simple_rnn_while_forward_simple_rnn_strided_slice_1_0"O
!forward_simple_rnn_while_identity*forward_simple_rnn/while/Identity:output:0"S
#forward_simple_rnn_while_identity_1,forward_simple_rnn/while/Identity_1:output:0"S
#forward_simple_rnn_while_identity_2,forward_simple_rnn/while/Identity_2:output:0"S
#forward_simple_rnn_while_identity_3,forward_simple_rnn/while/Identity_3:output:0"S
#forward_simple_rnn_while_identity_4,forward_simple_rnn/while/Identity_4:output:0"?
Jforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourceLforward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
Kforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resourceMforward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
Iforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceKforward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
wforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensoryforward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Aforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpAforward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp@forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Bforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpBforward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_378397
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_378397___redundant_placeholder04
0while_while_cond_378397___redundant_placeholder14
0while_while_cond_378397___redundant_placeholder24
0while_while_cond_378397___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_379041

inputs

states0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????
2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityg

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_namestates
?
?
while_cond_383370
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_383370___redundant_placeholder04
0while_while_cond_383370___redundant_placeholder14
0while_while_cond_383370___redundant_placeholder24
0while_while_cond_383370___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?>
?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_378461

inputs*
simple_rnn_cell_1_378386:
&
simple_rnn_cell_1_378388:
*
simple_rnn_cell_1_378390:


identity??)simple_rnn_cell_1/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
)simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_1_378386simple_rnn_cell_1_378388simple_rnn_cell_1_378390*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_3783852+
)simple_rnn_cell_1/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_1_378386simple_rnn_cell_1_378388simple_rnn_cell_1_378390*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_378398*
condR
while_cond_378397*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp*^simple_rnn_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2V
)simple_rnn_cell_1/StatefulPartitionedCall)simple_rnn_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_379734
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_379734___redundant_placeholder04
0while_while_cond_379734___redundant_placeholder14
0while_while_cond_379734___redundant_placeholder24
0while_while_cond_379734___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?#
?
while_body_379110
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_2_379132_0:
.
 while_simple_rnn_cell_2_379134_0:
2
 while_simple_rnn_cell_2_379136_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_2_379132:
,
while_simple_rnn_cell_2_379134:
0
while_simple_rnn_cell_2_379136:

??/while/simple_rnn_cell_2/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
/while/simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_2_379132_0 while_simple_rnn_cell_2_379134_0 while_simple_rnn_cell_2_379136_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_37904121
/while/simple_rnn_cell_2/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp0^while/simple_rnn_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_2_379132 while_simple_rnn_cell_2_379132_0"B
while_simple_rnn_cell_2_379134 while_simple_rnn_cell_2_379134_0"B
while_simple_rnn_cell_2_379136 while_simple_rnn_cell_2_379136_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2b
/while/simple_rnn_cell_2/StatefulPartitionedCall/while/simple_rnn_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?L
?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383665

inputsB
0simple_rnn_cell_2_matmul_readvariableop_resource:
?
1simple_rnn_cell_2_biasadd_readvariableop_resource:
D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	ReverseV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_383599*
condR
while_cond_383598*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
.__inference_bidirectional_layer_call_fn_381689

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_bidirectional_layer_call_and_return_conditional_losses_3803192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?e
?
__inference__traced_save_383966
file_prefix'
#savev2_variable_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopX
Tsavev2_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_read_readvariableopb
^savev2_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_read_readvariableopV
Rsavev2_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_read_readvariableopY
Usavev2_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_read_readvariableopc
_savev2_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_read_readvariableopW
Ssavev2_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop.
*savev2_adam_variable_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop_
[savev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_m_read_readvariableopi
esavev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_m_read_readvariableop]
Ysavev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_m_read_readvariableop`
\savev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_m_read_readvariableopj
fsavev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_m_read_readvariableop^
Zsavev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_m_read_readvariableop.
*savev2_adam_variable_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop_
[savev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_v_read_readvariableopi
esavev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_v_read_readvariableop]
Ysavev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_v_read_readvariableop`
\savev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_v_read_readvariableopj
fsavev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_v_read_readvariableop^
Zsavev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_v_read_readvariableop
savev2_const_1

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*?
value?B?-B:layer_with_weights-0/Variable:0/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/Variable:0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/Variable:0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopTsavev2_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_read_readvariableop^savev2_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_read_readvariableopRsavev2_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_read_readvariableopUsavev2_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_read_readvariableop_savev2_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_read_readvariableopSsavev2_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop*savev2_adam_variable_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop[savev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_m_read_readvariableopesavev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_m_read_readvariableopYsavev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_m_read_readvariableop\savev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_m_read_readvariableopfsavev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_m_read_readvariableopZsavev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_m_read_readvariableop*savev2_adam_variable_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop[savev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_kernel_v_read_readvariableopesavev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_recurrent_kernel_v_read_readvariableopYsavev2_adam_bidirectional_forward_simple_rnn_simple_rnn_cell_1_bias_v_read_readvariableop\savev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_kernel_v_read_readvariableopfsavev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_recurrent_kernel_v_read_readvariableopZsavev2_adam_bidirectional_backward_simple_rnn_simple_rnn_cell_2_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??::::: : : : : :
:

:
:
:

:
: : : : : : :
??:::::
:

:
:
:

:
:
??:::::
:

:
:
:

:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
:$ 

_output_shapes

:

: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
:$  

_output_shapes

:

: !

_output_shapes
:
:&""
 
_output_shapes
:
??:$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:
:$( 

_output_shapes

:

: )

_output_shapes
:
:$* 

_output_shapes

:
:$+ 

_output_shapes

:

: ,

_output_shapes
:
:-

_output_shapes
: 
??
?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_381310

inputs
embed_381054
embed_381056	 
embed_381058:
??c
Qbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
`
Rbidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
e
Sbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

d
Rbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
a
Sbidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
f
Tbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:

8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??Jbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Ibidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?Kbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?'bidirectional/backward_simple_rnn/while?Ibidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?Hbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?Jbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?&bidirectional/forward_simple_rnn/while?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embed/StatefulPartitionedCall?
embed/StatefulPartitionedCallStatefulPartitionedCallinputsembed_381054embed_381056embed_381058*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_2217052
embed/StatefulPartitionedCallt
reshape/ShapeShape&embed/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape&embed/StatefulPartitionedCall:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape/Reshape?
&bidirectional/forward_simple_rnn/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2(
&bidirectional/forward_simple_rnn/Shape?
4bidirectional/forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4bidirectional/forward_simple_rnn/strided_slice/stack?
6bidirectional/forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6bidirectional/forward_simple_rnn/strided_slice/stack_1?
6bidirectional/forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6bidirectional/forward_simple_rnn/strided_slice/stack_2?
.bidirectional/forward_simple_rnn/strided_sliceStridedSlice/bidirectional/forward_simple_rnn/Shape:output:0=bidirectional/forward_simple_rnn/strided_slice/stack:output:0?bidirectional/forward_simple_rnn/strided_slice/stack_1:output:0?bidirectional/forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.bidirectional/forward_simple_rnn/strided_slice?
,bidirectional/forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2.
,bidirectional/forward_simple_rnn/zeros/mul/y?
*bidirectional/forward_simple_rnn/zeros/mulMul7bidirectional/forward_simple_rnn/strided_slice:output:05bidirectional/forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2,
*bidirectional/forward_simple_rnn/zeros/mul?
-bidirectional/forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2/
-bidirectional/forward_simple_rnn/zeros/Less/y?
+bidirectional/forward_simple_rnn/zeros/LessLess.bidirectional/forward_simple_rnn/zeros/mul:z:06bidirectional/forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2-
+bidirectional/forward_simple_rnn/zeros/Less?
/bidirectional/forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
21
/bidirectional/forward_simple_rnn/zeros/packed/1?
-bidirectional/forward_simple_rnn/zeros/packedPack7bidirectional/forward_simple_rnn/strided_slice:output:08bidirectional/forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2/
-bidirectional/forward_simple_rnn/zeros/packed?
,bidirectional/forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,bidirectional/forward_simple_rnn/zeros/Const?
&bidirectional/forward_simple_rnn/zerosFill6bidirectional/forward_simple_rnn/zeros/packed:output:05bidirectional/forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2(
&bidirectional/forward_simple_rnn/zeros?
/bidirectional/forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          21
/bidirectional/forward_simple_rnn/transpose/perm?
*bidirectional/forward_simple_rnn/transpose	Transposereshape/Reshape:output:08bidirectional/forward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2,
*bidirectional/forward_simple_rnn/transpose?
(bidirectional/forward_simple_rnn/Shape_1Shape.bidirectional/forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2*
(bidirectional/forward_simple_rnn/Shape_1?
6bidirectional/forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6bidirectional/forward_simple_rnn/strided_slice_1/stack?
8bidirectional/forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_1/stack_1?
8bidirectional/forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_1/stack_2?
0bidirectional/forward_simple_rnn/strided_slice_1StridedSlice1bidirectional/forward_simple_rnn/Shape_1:output:0?bidirectional/forward_simple_rnn/strided_slice_1/stack:output:0Abidirectional/forward_simple_rnn/strided_slice_1/stack_1:output:0Abidirectional/forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0bidirectional/forward_simple_rnn/strided_slice_1?
<bidirectional/forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<bidirectional/forward_simple_rnn/TensorArrayV2/element_shape?
.bidirectional/forward_simple_rnn/TensorArrayV2TensorListReserveEbidirectional/forward_simple_rnn/TensorArrayV2/element_shape:output:09bidirectional/forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.bidirectional/forward_simple_rnn/TensorArrayV2?
Vbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2X
Vbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
Hbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor.bidirectional/forward_simple_rnn/transpose:y:0_bidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02J
Hbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
6bidirectional/forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6bidirectional/forward_simple_rnn/strided_slice_2/stack?
8bidirectional/forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_2/stack_1?
8bidirectional/forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_2/stack_2?
0bidirectional/forward_simple_rnn/strided_slice_2StridedSlice.bidirectional/forward_simple_rnn/transpose:y:0?bidirectional/forward_simple_rnn/strided_slice_2/stack:output:0Abidirectional/forward_simple_rnn/strided_slice_2/stack_1:output:0Abidirectional/forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask22
0bidirectional/forward_simple_rnn/strided_slice_2?
Hbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpQbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02J
Hbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
9bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul9bidirectional/forward_simple_rnn/strided_slice_2:output:0Pbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2;
9bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul?
Ibidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpRbidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02K
Ibidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
:bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAddCbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0Qbidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2<
:bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
Jbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpSbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02L
Jbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
;bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMul/bidirectional/forward_simple_rnn/zeros:output:0Rbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2=
;bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
6bidirectional/forward_simple_rnn/simple_rnn_cell_1/addAddV2Cbidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:0Ebidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
28
6bidirectional/forward_simple_rnn/simple_rnn_cell_1/add?
7bidirectional/forward_simple_rnn/simple_rnn_cell_1/TanhTanh:bidirectional/forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
29
7bidirectional/forward_simple_rnn/simple_rnn_cell_1/Tanh?
>bidirectional/forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2@
>bidirectional/forward_simple_rnn/TensorArrayV2_1/element_shape?
0bidirectional/forward_simple_rnn/TensorArrayV2_1TensorListReserveGbidirectional/forward_simple_rnn/TensorArrayV2_1/element_shape:output:09bidirectional/forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0bidirectional/forward_simple_rnn/TensorArrayV2_1?
%bidirectional/forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2'
%bidirectional/forward_simple_rnn/time?
9bidirectional/forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2;
9bidirectional/forward_simple_rnn/while/maximum_iterations?
3bidirectional/forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 25
3bidirectional/forward_simple_rnn/while/loop_counter?	
&bidirectional/forward_simple_rnn/whileWhile<bidirectional/forward_simple_rnn/while/loop_counter:output:0Bbidirectional/forward_simple_rnn/while/maximum_iterations:output:0.bidirectional/forward_simple_rnn/time:output:09bidirectional/forward_simple_rnn/TensorArrayV2_1:handle:0/bidirectional/forward_simple_rnn/zeros:output:09bidirectional/forward_simple_rnn/strided_slice_1:output:0Xbidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Qbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourceRbidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourceSbidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *>
body6R4
2bidirectional_forward_simple_rnn_while_body_381113*>
cond6R4
2bidirectional_forward_simple_rnn_while_cond_381112*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2(
&bidirectional/forward_simple_rnn/while?
Qbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2S
Qbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
Cbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack/bidirectional/forward_simple_rnn/while:output:3Zbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02E
Cbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
6bidirectional/forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6bidirectional/forward_simple_rnn/strided_slice_3/stack?
8bidirectional/forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8bidirectional/forward_simple_rnn/strided_slice_3/stack_1?
8bidirectional/forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8bidirectional/forward_simple_rnn/strided_slice_3/stack_2?
0bidirectional/forward_simple_rnn/strided_slice_3StridedSliceLbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0?bidirectional/forward_simple_rnn/strided_slice_3/stack:output:0Abidirectional/forward_simple_rnn/strided_slice_3/stack_1:output:0Abidirectional/forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask22
0bidirectional/forward_simple_rnn/strided_slice_3?
1bidirectional/forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          23
1bidirectional/forward_simple_rnn/transpose_1/perm?
,bidirectional/forward_simple_rnn/transpose_1	TransposeLbidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0:bidirectional/forward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2.
,bidirectional/forward_simple_rnn/transpose_1?
'bidirectional/backward_simple_rnn/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2)
'bidirectional/backward_simple_rnn/Shape?
5bidirectional/backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5bidirectional/backward_simple_rnn/strided_slice/stack?
7bidirectional/backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7bidirectional/backward_simple_rnn/strided_slice/stack_1?
7bidirectional/backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7bidirectional/backward_simple_rnn/strided_slice/stack_2?
/bidirectional/backward_simple_rnn/strided_sliceStridedSlice0bidirectional/backward_simple_rnn/Shape:output:0>bidirectional/backward_simple_rnn/strided_slice/stack:output:0@bidirectional/backward_simple_rnn/strided_slice/stack_1:output:0@bidirectional/backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/bidirectional/backward_simple_rnn/strided_slice?
-bidirectional/backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2/
-bidirectional/backward_simple_rnn/zeros/mul/y?
+bidirectional/backward_simple_rnn/zeros/mulMul8bidirectional/backward_simple_rnn/strided_slice:output:06bidirectional/backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2-
+bidirectional/backward_simple_rnn/zeros/mul?
.bidirectional/backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?20
.bidirectional/backward_simple_rnn/zeros/Less/y?
,bidirectional/backward_simple_rnn/zeros/LessLess/bidirectional/backward_simple_rnn/zeros/mul:z:07bidirectional/backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2.
,bidirectional/backward_simple_rnn/zeros/Less?
0bidirectional/backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
22
0bidirectional/backward_simple_rnn/zeros/packed/1?
.bidirectional/backward_simple_rnn/zeros/packedPack8bidirectional/backward_simple_rnn/strided_slice:output:09bidirectional/backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.bidirectional/backward_simple_rnn/zeros/packed?
-bidirectional/backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-bidirectional/backward_simple_rnn/zeros/Const?
'bidirectional/backward_simple_rnn/zerosFill7bidirectional/backward_simple_rnn/zeros/packed:output:06bidirectional/backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2)
'bidirectional/backward_simple_rnn/zeros?
0bidirectional/backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          22
0bidirectional/backward_simple_rnn/transpose/perm?
+bidirectional/backward_simple_rnn/transpose	Transposereshape/Reshape:output:09bidirectional/backward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2-
+bidirectional/backward_simple_rnn/transpose?
)bidirectional/backward_simple_rnn/Shape_1Shape/bidirectional/backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2+
)bidirectional/backward_simple_rnn/Shape_1?
7bidirectional/backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7bidirectional/backward_simple_rnn/strided_slice_1/stack?
9bidirectional/backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_1/stack_1?
9bidirectional/backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_1/stack_2?
1bidirectional/backward_simple_rnn/strided_slice_1StridedSlice2bidirectional/backward_simple_rnn/Shape_1:output:0@bidirectional/backward_simple_rnn/strided_slice_1/stack:output:0Bbidirectional/backward_simple_rnn/strided_slice_1/stack_1:output:0Bbidirectional/backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1bidirectional/backward_simple_rnn/strided_slice_1?
=bidirectional/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2?
=bidirectional/backward_simple_rnn/TensorArrayV2/element_shape?
/bidirectional/backward_simple_rnn/TensorArrayV2TensorListReserveFbidirectional/backward_simple_rnn/TensorArrayV2/element_shape:output:0:bidirectional/backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/bidirectional/backward_simple_rnn/TensorArrayV2?
0bidirectional/backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 22
0bidirectional/backward_simple_rnn/ReverseV2/axis?
+bidirectional/backward_simple_rnn/ReverseV2	ReverseV2/bidirectional/backward_simple_rnn/transpose:y:09bidirectional/backward_simple_rnn/ReverseV2/axis:output:0*
T0*+
_output_shapes
:?????????2-
+bidirectional/backward_simple_rnn/ReverseV2?
Wbidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2Y
Wbidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
Ibidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor4bidirectional/backward_simple_rnn/ReverseV2:output:0`bidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02K
Ibidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
7bidirectional/backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7bidirectional/backward_simple_rnn/strided_slice_2/stack?
9bidirectional/backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_2/stack_1?
9bidirectional/backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_2/stack_2?
1bidirectional/backward_simple_rnn/strided_slice_2StridedSlice/bidirectional/backward_simple_rnn/transpose:y:0@bidirectional/backward_simple_rnn/strided_slice_2/stack:output:0Bbidirectional/backward_simple_rnn/strided_slice_2/stack_1:output:0Bbidirectional/backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask23
1bidirectional/backward_simple_rnn/strided_slice_2?
Ibidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpRbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02K
Ibidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
:bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul:bidirectional/backward_simple_rnn/strided_slice_2:output:0Qbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2<
:bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul?
Jbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpSbidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02L
Jbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
;bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAddDbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0Rbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2=
;bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
Kbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpTbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02M
Kbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
<bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMul0bidirectional/backward_simple_rnn/zeros:output:0Sbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2>
<bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
7bidirectional/backward_simple_rnn/simple_rnn_cell_2/addAddV2Dbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:0Fbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
29
7bidirectional/backward_simple_rnn/simple_rnn_cell_2/add?
8bidirectional/backward_simple_rnn/simple_rnn_cell_2/TanhTanh;bidirectional/backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2:
8bidirectional/backward_simple_rnn/simple_rnn_cell_2/Tanh?
?bidirectional/backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2A
?bidirectional/backward_simple_rnn/TensorArrayV2_1/element_shape?
1bidirectional/backward_simple_rnn/TensorArrayV2_1TensorListReserveHbidirectional/backward_simple_rnn/TensorArrayV2_1/element_shape:output:0:bidirectional/backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1bidirectional/backward_simple_rnn/TensorArrayV2_1?
&bidirectional/backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2(
&bidirectional/backward_simple_rnn/time?
:bidirectional/backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2<
:bidirectional/backward_simple_rnn/while/maximum_iterations?
4bidirectional/backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 26
4bidirectional/backward_simple_rnn/while/loop_counter?	
'bidirectional/backward_simple_rnn/whileWhile=bidirectional/backward_simple_rnn/while/loop_counter:output:0Cbidirectional/backward_simple_rnn/while/maximum_iterations:output:0/bidirectional/backward_simple_rnn/time:output:0:bidirectional/backward_simple_rnn/TensorArrayV2_1:handle:00bidirectional/backward_simple_rnn/zeros:output:0:bidirectional/backward_simple_rnn/strided_slice_1:output:0Ybidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Rbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceSbidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceTbidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *?
body7R5
3bidirectional_backward_simple_rnn_while_body_381223*?
cond7R5
3bidirectional_backward_simple_rnn_while_cond_381222*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2)
'bidirectional/backward_simple_rnn/while?
Rbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2T
Rbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
Dbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack0bidirectional/backward_simple_rnn/while:output:3[bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02F
Dbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
7bidirectional/backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????29
7bidirectional/backward_simple_rnn/strided_slice_3/stack?
9bidirectional/backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9bidirectional/backward_simple_rnn/strided_slice_3/stack_1?
9bidirectional/backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9bidirectional/backward_simple_rnn/strided_slice_3/stack_2?
1bidirectional/backward_simple_rnn/strided_slice_3StridedSliceMbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0@bidirectional/backward_simple_rnn/strided_slice_3/stack:output:0Bbidirectional/backward_simple_rnn/strided_slice_3/stack_1:output:0Bbidirectional/backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask23
1bidirectional/backward_simple_rnn/strided_slice_3?
2bidirectional/backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          24
2bidirectional/backward_simple_rnn/transpose_1/perm?
-bidirectional/backward_simple_rnn/transpose_1	TransposeMbidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0;bidirectional/backward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2/
-bidirectional/backward_simple_rnn/transpose_1x
bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
bidirectional/concat/axis?
bidirectional/concatConcatV29bidirectional/forward_simple_rnn/strided_slice_3:output:0:bidirectional/backward_simple_rnn/strided_slice_3:output:0"bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
bidirectional/concat?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulbidirectional/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu?
dropout_1/IdentityIdentitydense_2/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_1/Identity?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout_1/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Reluc
dense_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
dense_3/Constg
dense_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
dense_3/Const_1?
dense_3/clip_by_value/MinimumMinimumdense_3/Relu:activations:0dense_3/Const:output:0*
T0*'
_output_shapes
:?????????2
dense_3/clip_by_value/Minimum?
dense_3/clip_by_valueMaximum!dense_3/clip_by_value/Minimum:z:0dense_3/Const_1:output:0*
T0*'
_output_shapes
:?????????2
dense_3/clip_by_valuet
IdentityIdentitydense_3/clip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOpK^bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpJ^bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpL^bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp(^bidirectional/backward_simple_rnn/whileJ^bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpI^bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpK^bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp'^bidirectional/forward_simple_rnn/while^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 2?
Jbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpJbidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Ibidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpIbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Kbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpKbidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp2R
'bidirectional/backward_simple_rnn/while'bidirectional/backward_simple_rnn/while2?
Ibidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpIbidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
Hbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpHbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Jbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpJbidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp2P
&bidirectional/forward_simple_rnn/while&bidirectional/forward_simple_rnn/while2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
embed/StatefulPartitionedCallembed/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
2bidirectional_forward_simple_rnn_while_cond_381371^
Zbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_loop_counterd
`bidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_maximum_iterations6
2bidirectional_forward_simple_rnn_while_placeholder8
4bidirectional_forward_simple_rnn_while_placeholder_18
4bidirectional_forward_simple_rnn_while_placeholder_2`
\bidirectional_forward_simple_rnn_while_less_bidirectional_forward_simple_rnn_strided_slice_1v
rbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_cond_381371___redundant_placeholder0v
rbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_cond_381371___redundant_placeholder1v
rbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_cond_381371___redundant_placeholder2v
rbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_cond_381371___redundant_placeholder33
/bidirectional_forward_simple_rnn_while_identity
?
+bidirectional/forward_simple_rnn/while/LessLess2bidirectional_forward_simple_rnn_while_placeholder\bidirectional_forward_simple_rnn_while_less_bidirectional_forward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2-
+bidirectional/forward_simple_rnn/while/Less?
/bidirectional/forward_simple_rnn/while/IdentityIdentity/bidirectional/forward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 21
/bidirectional/forward_simple_rnn/while/Identity"k
/bidirectional_forward_simple_rnn_while_identity8bidirectional/forward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?K
?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383323
inputs_0B
0simple_rnn_cell_2_matmul_readvariableop_resource:
?
1simple_rnn_cell_2_biasadd_readvariableop_resource:
D
2simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_2/BiasAdd/ReadVariableOp?'simple_rnn_cell_2/MatMul/ReadVariableOp?)simple_rnn_cell_2/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????2
	ReverseV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_2/MatMul/ReadVariableOp?
simple_rnn_cell_2/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul?
(simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_2/BiasAdd/ReadVariableOp?
simple_rnn_cell_2/BiasAddBiasAdd"simple_rnn_cell_2/MatMul:product:00simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/BiasAdd?
)simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_2/MatMul_1/ReadVariableOp?
simple_rnn_cell_2/MatMul_1MatMulzeros:output:01simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/MatMul_1?
simple_rnn_cell_2/addAddV2"simple_rnn_cell_2/BiasAdd:output:0$simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/add?
simple_rnn_cell_2/TanhTanhsimple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_2/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_2_matmul_readvariableop_resource1simple_rnn_cell_2_biasadd_readvariableop_resource2simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_383257*
condR
while_cond_383256*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_2/BiasAdd/ReadVariableOp(^simple_rnn_cell_2/MatMul/ReadVariableOp*^simple_rnn_cell_2/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2T
(simple_rnn_cell_2/BiasAdd/ReadVariableOp(simple_rnn_cell_2/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_2/MatMul/ReadVariableOp'simple_rnn_cell_2/MatMul/ReadVariableOp2V
)simple_rnn_cell_2/MatMul_1/ReadVariableOp)simple_rnn_cell_2/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
C__inference_dense_3_layer_call_and_return_conditional_losses_380372

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
ReluS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1?
clip_by_value/MinimumMinimumRelu:activations:0Const:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimum?
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valuel
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_382874
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_382874___redundant_placeholder04
0while_while_cond_382874___redundant_placeholder14
0while_while_cond_382874___redundant_placeholder24
0while_while_cond_382874___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?G
?
%backward_simple_rnn_while_body_380251D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2C
?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0
{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
[
Mbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
`
Nbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:

&
"backward_simple_rnn_while_identity(
$backward_simple_rnn_while_identity_1(
$backward_simple_rnn_while_identity_2(
$backward_simple_rnn_while_identity_3(
$backward_simple_rnn_while_identity_4A
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1}
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor\
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource:
Y
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource:
^
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Kbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0%backward_simple_rnn_while_placeholderTbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02?
=backward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02C
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp?
2backward_simple_rnn/while/simple_rnn_cell_2/MatMulMatMulDbackward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Ibackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
24
2backward_simple_rnn/while/simple_rnn_cell_2/MatMul?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02D
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAddBiasAdd<backward_simple_rnn/while/simple_rnn_cell_2/MatMul:product:0Jbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
25
3backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02E
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1MatMul'backward_simple_rnn_while_placeholder_2Kbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
26
4backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1?
/backward_simple_rnn/while/simple_rnn_cell_2/addAddV2<backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd:output:0>backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
21
/backward_simple_rnn/while/simple_rnn_cell_2/add?
0backward_simple_rnn/while/simple_rnn_cell_2/TanhTanh3backward_simple_rnn/while/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
22
0backward_simple_rnn/while/simple_rnn_cell_2/Tanh?
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'backward_simple_rnn_while_placeholder_1%backward_simple_rnn_while_placeholder4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02@
>backward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
backward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
backward_simple_rnn/while/add/y?
backward_simple_rnn/while/addAddV2%backward_simple_rnn_while_placeholder(backward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/while/add?
!backward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!backward_simple_rnn/while/add_1/y?
backward_simple_rnn/while/add_1AddV2@backward_simple_rnn_while_backward_simple_rnn_while_loop_counter*backward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
backward_simple_rnn/while/add_1?
"backward_simple_rnn/while/IdentityIdentity#backward_simple_rnn/while/add_1:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity?
$backward_simple_rnn/while/Identity_1IdentityFbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_1?
$backward_simple_rnn/while/Identity_2Identity!backward_simple_rnn/while/add:z:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_2?
$backward_simple_rnn/while/Identity_3IdentityNbackward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 2&
$backward_simple_rnn/while/Identity_3?
$backward_simple_rnn/while/Identity_4Identity4backward_simple_rnn/while/simple_rnn_cell_2/Tanh:y:0^backward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
2&
$backward_simple_rnn/while/Identity_4?
backward_simple_rnn/while/NoOpNoOpC^backward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpB^backward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpD^backward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
backward_simple_rnn/while/NoOp"?
=backward_simple_rnn_while_backward_simple_rnn_strided_slice_1?backward_simple_rnn_while_backward_simple_rnn_strided_slice_1_0"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0"U
$backward_simple_rnn_while_identity_1-backward_simple_rnn/while/Identity_1:output:0"U
$backward_simple_rnn_while_identity_2-backward_simple_rnn/while/Identity_2:output:0"U
$backward_simple_rnn_while_identity_3-backward_simple_rnn/while/Identity_3:output:0"U
$backward_simple_rnn_while_identity_4-backward_simple_rnn/while/Identity_4:output:0"?
Kbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resourceMbackward_simple_rnn_while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"?
Lbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resourceNbackward_simple_rnn_while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"?
Jbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resourceLbackward_simple_rnn_while_simple_rnn_cell_2_matmul_readvariableop_resource_0"?
ybackward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor{backward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_backward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Bbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOpBbackward_simple_rnn/while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
Abackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOpAbackward_simple_rnn/while/simple_rnn_cell_2/MatMul/ReadVariableOp2?
Cbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpCbackward_simple_rnn/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_383789

inputs
states_00
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????
2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityg

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????

"
_user_specified_name
states/0
?	
?
.__inference_bidirectional_layer_call_fn_381672
inputs_0
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:


identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_bidirectional_layer_call_and_return_conditional_losses_3799832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'???????????????????????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/0
?1
?
while_body_379735
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_380438

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_382602

inputsU
Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
R
Dforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
W
Eforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

V
Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
S
Ebackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
X
Fbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:


identity??<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?backward_simple_rnn/while?;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?forward_simple_rnn/whilej
forward_simple_rnn/ShapeShapeinputs*
T0*
_output_shapes
:2
forward_simple_rnn/Shape?
&forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_simple_rnn/strided_slice/stack?
(forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_1?
(forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(forward_simple_rnn/strided_slice/stack_2?
 forward_simple_rnn/strided_sliceStridedSlice!forward_simple_rnn/Shape:output:0/forward_simple_rnn/strided_slice/stack:output:01forward_simple_rnn/strided_slice/stack_1:output:01forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 forward_simple_rnn/strided_slice?
forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2 
forward_simple_rnn/zeros/mul/y?
forward_simple_rnn/zeros/mulMul)forward_simple_rnn/strided_slice:output:0'forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/mul?
forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
forward_simple_rnn/zeros/Less/y?
forward_simple_rnn/zeros/LessLess forward_simple_rnn/zeros/mul:z:0(forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_simple_rnn/zeros/Less?
!forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2#
!forward_simple_rnn/zeros/packed/1?
forward_simple_rnn/zeros/packedPack)forward_simple_rnn/strided_slice:output:0*forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
forward_simple_rnn/zeros/packed?
forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
forward_simple_rnn/zeros/Const?
forward_simple_rnn/zerosFill(forward_simple_rnn/zeros/packed:output:0'forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
forward_simple_rnn/zeros?
!forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!forward_simple_rnn/transpose/perm?
forward_simple_rnn/transpose	Transposeinputs*forward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
forward_simple_rnn/transpose?
forward_simple_rnn/Shape_1Shape forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
forward_simple_rnn/Shape_1?
(forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_1/stack?
*forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_1?
*forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_1/stack_2?
"forward_simple_rnn/strided_slice_1StridedSlice#forward_simple_rnn/Shape_1:output:01forward_simple_rnn/strided_slice_1/stack:output:03forward_simple_rnn/strided_slice_1/stack_1:output:03forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_1?
.forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.forward_simple_rnn/TensorArrayV2/element_shape?
 forward_simple_rnn/TensorArrayV2TensorListReserve7forward_simple_rnn/TensorArrayV2/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 forward_simple_rnn/TensorArrayV2?
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2J
Hforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor forward_simple_rnn/transpose:y:0Qforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
(forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(forward_simple_rnn/strided_slice_2/stack?
*forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_1?
*forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_2/stack_2?
"forward_simple_rnn/strided_slice_2StridedSlice forward_simple_rnn/transpose:y:01forward_simple_rnn/strided_slice_2/stack:output:03forward_simple_rnn/strided_slice_2/stack_1:output:03forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_2?
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpCforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02<
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
+forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul+forward_simple_rnn/strided_slice_2:output:0Bforward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2-
+forward_simple_rnn/simple_rnn_cell_1/MatMul?
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02=
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
,forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAdd5forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0Cforward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02>
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMul!forward_simple_rnn/zeros:output:0Dforward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
(forward_simple_rnn/simple_rnn_cell_1/addAddV25forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:07forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2*
(forward_simple_rnn/simple_rnn_cell_1/add?
)forward_simple_rnn/simple_rnn_cell_1/TanhTanh,forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2+
)forward_simple_rnn/simple_rnn_cell_1/Tanh?
0forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0forward_simple_rnn/TensorArrayV2_1/element_shape?
"forward_simple_rnn/TensorArrayV2_1TensorListReserve9forward_simple_rnn/TensorArrayV2_1/element_shape:output:0+forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"forward_simple_rnn/TensorArrayV2_1t
forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_simple_rnn/time?
+forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+forward_simple_rnn/while/maximum_iterations?
%forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%forward_simple_rnn/while/loop_counter?
forward_simple_rnn/whileWhile.forward_simple_rnn/while/loop_counter:output:04forward_simple_rnn/while/maximum_iterations:output:0 forward_simple_rnn/time:output:0+forward_simple_rnn/TensorArrayV2_1:handle:0!forward_simple_rnn/zeros:output:0+forward_simple_rnn/strided_slice_1:output:0Jforward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Cforward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourceDforward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourceEforward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$forward_simple_rnn_while_body_382424*0
cond(R&
$forward_simple_rnn_while_cond_382423*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
forward_simple_rnn/while?
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2E
Cforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
5forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack!forward_simple_rnn/while:output:3Lforward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype027
5forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
(forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(forward_simple_rnn/strided_slice_3/stack?
*forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*forward_simple_rnn/strided_slice_3/stack_1?
*forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*forward_simple_rnn/strided_slice_3/stack_2?
"forward_simple_rnn/strided_slice_3StridedSlice>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:01forward_simple_rnn/strided_slice_3/stack:output:03forward_simple_rnn/strided_slice_3/stack_1:output:03forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2$
"forward_simple_rnn/strided_slice_3?
#forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#forward_simple_rnn/transpose_1/perm?
forward_simple_rnn/transpose_1	Transpose>forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0,forward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2 
forward_simple_rnn/transpose_1l
backward_simple_rnn/ShapeShapeinputs*
T0*
_output_shapes
:2
backward_simple_rnn/Shape?
'backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_simple_rnn/strided_slice/stack?
)backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_1?
)backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)backward_simple_rnn/strided_slice/stack_2?
!backward_simple_rnn/strided_sliceStridedSlice"backward_simple_rnn/Shape:output:00backward_simple_rnn/strided_slice/stack:output:02backward_simple_rnn/strided_slice/stack_1:output:02backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!backward_simple_rnn/strided_slice?
backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2!
backward_simple_rnn/zeros/mul/y?
backward_simple_rnn/zeros/mulMul*backward_simple_rnn/strided_slice:output:0(backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_simple_rnn/zeros/mul?
 backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 backward_simple_rnn/zeros/Less/y?
backward_simple_rnn/zeros/LessLess!backward_simple_rnn/zeros/mul:z:0)backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
backward_simple_rnn/zeros/Less?
"backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2$
"backward_simple_rnn/zeros/packed/1?
 backward_simple_rnn/zeros/packedPack*backward_simple_rnn/strided_slice:output:0+backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 backward_simple_rnn/zeros/packed?
backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
backward_simple_rnn/zeros/Const?
backward_simple_rnn/zerosFill)backward_simple_rnn/zeros/packed:output:0(backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
backward_simple_rnn/zeros?
"backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"backward_simple_rnn/transpose/perm?
backward_simple_rnn/transpose	Transposeinputs+backward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
backward_simple_rnn/transpose?
backward_simple_rnn/Shape_1Shape!backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2
backward_simple_rnn/Shape_1?
)backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_1/stack?
+backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_1?
+backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_1/stack_2?
#backward_simple_rnn/strided_slice_1StridedSlice$backward_simple_rnn/Shape_1:output:02backward_simple_rnn/strided_slice_1/stack:output:04backward_simple_rnn/strided_slice_1/stack_1:output:04backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_1?
/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/backward_simple_rnn/TensorArrayV2/element_shape?
!backward_simple_rnn/TensorArrayV2TensorListReserve8backward_simple_rnn/TensorArrayV2/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!backward_simple_rnn/TensorArrayV2?
"backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2$
"backward_simple_rnn/ReverseV2/axis?
backward_simple_rnn/ReverseV2	ReverseV2!backward_simple_rnn/transpose:y:0+backward_simple_rnn/ReverseV2/axis:output:0*
T0*+
_output_shapes
:?????????2
backward_simple_rnn/ReverseV2?
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2K
Ibackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor&backward_simple_rnn/ReverseV2:output:0Rbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
)backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)backward_simple_rnn/strided_slice_2/stack?
+backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_1?
+backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_2/stack_2?
#backward_simple_rnn/strided_slice_2StridedSlice!backward_simple_rnn/transpose:y:02backward_simple_rnn/strided_slice_2/stack:output:04backward_simple_rnn/strided_slice_2/stack_1:output:04backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_2?
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOpDbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02=
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
,backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul,backward_simple_rnn/strided_slice_2:output:0Cbackward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2.
,backward_simple_rnn/simple_rnn_cell_2/MatMul?
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02>
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
-backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAdd6backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0Dbackward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2/
-backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOpFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02?
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMul"backward_simple_rnn/zeros:output:0Ebackward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
20
.backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
)backward_simple_rnn/simple_rnn_cell_2/addAddV26backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:08backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2+
)backward_simple_rnn/simple_rnn_cell_2/add?
*backward_simple_rnn/simple_rnn_cell_2/TanhTanh-backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2,
*backward_simple_rnn/simple_rnn_cell_2/Tanh?
1backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   23
1backward_simple_rnn/TensorArrayV2_1/element_shape?
#backward_simple_rnn/TensorArrayV2_1TensorListReserve:backward_simple_rnn/TensorArrayV2_1/element_shape:output:0,backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#backward_simple_rnn/TensorArrayV2_1v
backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_simple_rnn/time?
,backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,backward_simple_rnn/while/maximum_iterations?
&backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&backward_simple_rnn/while/loop_counter?
backward_simple_rnn/whileWhile/backward_simple_rnn/while/loop_counter:output:05backward_simple_rnn/while/maximum_iterations:output:0!backward_simple_rnn/time:output:0,backward_simple_rnn/TensorArrayV2_1:handle:0"backward_simple_rnn/zeros:output:0,backward_simple_rnn/strided_slice_1:output:0Kbackward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dbackward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourceEbackward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourceFbackward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%backward_simple_rnn_while_body_382534*1
cond)R'
%backward_simple_rnn_while_cond_382533*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
backward_simple_rnn/while?
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2F
Dbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
6backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStack"backward_simple_rnn/while:output:3Mbackward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype028
6backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
)backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)backward_simple_rnn/strided_slice_3/stack?
+backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+backward_simple_rnn/strided_slice_3/stack_1?
+backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+backward_simple_rnn/strided_slice_3/stack_2?
#backward_simple_rnn/strided_slice_3StridedSlice?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:02backward_simple_rnn/strided_slice_3/stack:output:04backward_simple_rnn/strided_slice_3/stack_1:output:04backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2%
#backward_simple_rnn/strided_slice_3?
$backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$backward_simple_rnn/transpose_1/perm?
backward_simple_rnn/transpose_1	Transpose?backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0-backward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2!
backward_simple_rnn/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2+forward_simple_rnn/strided_slice_3:output:0,backward_simple_rnn/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatj
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp=^backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<^backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp>^backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp^backward_simple_rnn/while<^forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;^forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp=^forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp^forward_simple_rnn/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2|
<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp<backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2z
;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp;backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2~
=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp=backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp26
backward_simple_rnn/whilebackward_simple_rnn/while2z
;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp;forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2x
:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2|
<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp<forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp24
forward_simple_rnn/whileforward_simple_rnn/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_378385

inputs

states0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????
2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityg

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_namestates
?
8
__inference__creator_128
identity??
hash_table?

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*?
shared_name??hash_table_<google3.third_party.tensorflow.python.training.tracking.tracking.TrackableAsset object at 0x7faef27d6ed0>_-2_-1_load_35_124*
use_node_name_sharing(*
value_dtype0	2

hash_tablei
IdentityIdentityhash_table:table_handle:0^hash_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_381638

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?@
?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_378997

inputs*
simple_rnn_cell_2_378922:
&
simple_rnn_cell_2_378924:
*
simple_rnn_cell_2_378926:


identity??)simple_rnn_cell_2/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis?
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :??????????????????2
	ReverseV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
)simple_rnn_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_2_378922simple_rnn_cell_2_378924simple_rnn_cell_2_378926*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_3789212+
)simple_rnn_cell_2/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_2_378922simple_rnn_cell_2_378924simple_rnn_cell_2_378926*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_378934*
condR
while_cond_378933*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp*^simple_rnn_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2V
)simple_rnn_cell_2/StatefulPartitionedCall)simple_rnn_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?U
?
2bidirectional_forward_simple_rnn_while_body_381372^
Zbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_loop_counterd
`bidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_maximum_iterations6
2bidirectional_forward_simple_rnn_while_placeholder8
4bidirectional_forward_simple_rnn_while_placeholder_18
4bidirectional_forward_simple_rnn_while_placeholder_2]
Ybidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_strided_slice_1_0?
?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0k
Ybidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0:
h
Zbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0:
m
[bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0:

3
/bidirectional_forward_simple_rnn_while_identity5
1bidirectional_forward_simple_rnn_while_identity_15
1bidirectional_forward_simple_rnn_while_identity_25
1bidirectional_forward_simple_rnn_while_identity_35
1bidirectional_forward_simple_rnn_while_identity_4[
Wbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_strided_slice_1?
?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensori
Wbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource:
f
Xbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource:
k
Ybidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource:

??Obidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?Nbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?Pbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Xbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2Z
Xbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Jbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_02bidirectional_forward_simple_rnn_while_placeholderabidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02L
Jbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem?
Nbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpYbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02P
Nbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp?
?bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMulMatMulQbidirectional/forward_simple_rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Vbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2A
?bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul?
Obidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOpZbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype02Q
Obidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
@bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAddBiasAddIbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul:product:0Wbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2B
@bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd?
Pbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp[bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype02R
Pbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
Abidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1MatMul4bidirectional_forward_simple_rnn_while_placeholder_2Xbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2C
Abidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1?
<bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/addAddV2Ibidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd:output:0Kbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2>
<bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/add?
=bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/TanhTanh@bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2?
=bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh?
Kbidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem4bidirectional_forward_simple_rnn_while_placeholder_12bidirectional_forward_simple_rnn_while_placeholderAbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0*
_output_shapes
: *
element_dtype02M
Kbidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem?
,bidirectional/forward_simple_rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,bidirectional/forward_simple_rnn/while/add/y?
*bidirectional/forward_simple_rnn/while/addAddV22bidirectional_forward_simple_rnn_while_placeholder5bidirectional/forward_simple_rnn/while/add/y:output:0*
T0*
_output_shapes
: 2,
*bidirectional/forward_simple_rnn/while/add?
.bidirectional/forward_simple_rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.bidirectional/forward_simple_rnn/while/add_1/y?
,bidirectional/forward_simple_rnn/while/add_1AddV2Zbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_loop_counter7bidirectional/forward_simple_rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2.
,bidirectional/forward_simple_rnn/while/add_1?
/bidirectional/forward_simple_rnn/while/IdentityIdentity0bidirectional/forward_simple_rnn/while/add_1:z:0,^bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 21
/bidirectional/forward_simple_rnn/while/Identity?
1bidirectional/forward_simple_rnn/while/Identity_1Identity`bidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_while_maximum_iterations,^bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 23
1bidirectional/forward_simple_rnn/while/Identity_1?
1bidirectional/forward_simple_rnn/while/Identity_2Identity.bidirectional/forward_simple_rnn/while/add:z:0,^bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 23
1bidirectional/forward_simple_rnn/while/Identity_2?
1bidirectional/forward_simple_rnn/while/Identity_3Identity[bidirectional/forward_simple_rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^bidirectional/forward_simple_rnn/while/NoOp*
T0*
_output_shapes
: 23
1bidirectional/forward_simple_rnn/while/Identity_3?
1bidirectional/forward_simple_rnn/while/Identity_4IdentityAbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/Tanh:y:0,^bidirectional/forward_simple_rnn/while/NoOp*
T0*'
_output_shapes
:?????????
23
1bidirectional/forward_simple_rnn/while/Identity_4?
+bidirectional/forward_simple_rnn/while/NoOpNoOpP^bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpO^bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpQ^bidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2-
+bidirectional/forward_simple_rnn/while/NoOp"?
Wbidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_strided_slice_1Ybidirectional_forward_simple_rnn_while_bidirectional_forward_simple_rnn_strided_slice_1_0"k
/bidirectional_forward_simple_rnn_while_identity8bidirectional/forward_simple_rnn/while/Identity:output:0"o
1bidirectional_forward_simple_rnn_while_identity_1:bidirectional/forward_simple_rnn/while/Identity_1:output:0"o
1bidirectional_forward_simple_rnn_while_identity_2:bidirectional/forward_simple_rnn/while/Identity_2:output:0"o
1bidirectional_forward_simple_rnn_while_identity_3:bidirectional/forward_simple_rnn/while/Identity_3:output:0"o
1bidirectional_forward_simple_rnn_while_identity_4:bidirectional/forward_simple_rnn/while/Identity_4:output:0"?
Xbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resourceZbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_biasadd_readvariableop_resource_0"?
Ybidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource[bidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_1_readvariableop_resource_0"?
Wbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resourceYbidirectional_forward_simple_rnn_while_simple_rnn_cell_1_matmul_readvariableop_resource_0"?
?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor?bidirectional_forward_simple_rnn_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_simple_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2?
Obidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOpObidirectional/forward_simple_rnn/while/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
Nbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOpNbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul/ReadVariableOp2?
Pbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOpPbidirectional/forward_simple_rnn/while/simple_rnn_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
??
?
!__inference__wrapped_model_378333	
input0
,rnn_reg_bidirect_swivel_balance_embed_3780770
,rnn_reg_bidirect_swivel_balance_embed_378079	@
,rnn_reg_bidirect_swivel_balance_embed_378081:
???
qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource:
?
rrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource:
?
srnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource:

?
rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource:
?
srnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource:
?
trnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource:

X
Frnn_reg_bidirect_swivel_balance_dense_2_matmul_readvariableop_resource:U
Grnn_reg_bidirect_swivel_balance_dense_2_biasadd_readvariableop_resource:X
Frnn_reg_bidirect_swivel_balance_dense_3_matmul_readvariableop_resource:U
Grnn_reg_bidirect_swivel_balance_dense_3_biasadd_readvariableop_resource:
identity??n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?K2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while?m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?l2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while?B2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd/ReadVariableOp?A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul/ReadVariableOp?B2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd/ReadVariableOp?A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul/ReadVariableOp?A2.3-RNN-Reg-Bidirect-Swivel-Balance/embed/StatefulPartitionedCall?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/embed/StatefulPartitionedCallStatefulPartitionedCallinput,rnn_reg_bidirect_swivel_balance_embed_378077,rnn_reg_bidirect_swivel_balance_embed_378079,rnn_reg_bidirect_swivel_balance_embed_378081*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_2217052C
A2.3-RNN-Reg-Bidirect-Swivel-Balance/embed/StatefulPartitionedCall?
12.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/ShapeShapeJ2.3-RNN-Reg-Bidirect-Swivel-Balance/embed/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:23
12.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Shape?
?2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stack?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
A2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stack_1?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
A2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stack_2?
92.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_sliceStridedSlice:2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Shape:output:0H2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stack:output:0J2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stack_1:output:0J2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
92.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice?
;2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shape/1?
;2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2=
;2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shape/2?
92.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shapePackB2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/strided_slice:output:0D2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shape/1:output:0D2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2;
92.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shape?
32.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/ReshapeReshapeJ2.3-RNN-Reg-Bidirect-Swivel-Balance/embed/StatefulPartitionedCall:output:0B2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????25
32.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape?
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/ShapeShape<2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape:output:0*
T0*
_output_shapes
:2L
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/Shape?
X2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Z
X2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stack?
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2\
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stack_1?
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2\
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stack_2?
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_sliceStridedSliceS2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/Shape:output:0a2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stack:output:0c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stack_1:output:0c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2T
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice?
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2R
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/mul/y?
N2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/mulMul[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice:output:0Y2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2P
N2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/mul?
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2S
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/Less/y?
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/LessLessR2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/mul:z:0Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2Q
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/Less?
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2U
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/packed/1?
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/packedPack[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice:output:0\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2S
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/packed?
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2R
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/Const?
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zerosFillZ2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/packed:output:0Y2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2L
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros?
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2U
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose/perm?
N2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose	Transpose<2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape:output:0\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2P
N2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose?
L2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/Shape_1ShapeR2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2N
L2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/Shape_1?
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2\
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stack?
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stack_1?
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stack_2?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1StridedSliceU2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/Shape_1:output:0c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stack:output:0e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stack_1:output:0e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1?
`2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2b
`2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2/element_shape?
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2TensorListReservei2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2/element_shape:output:0]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02T
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2?
z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2|
z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
l2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorR2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose:y:0?2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02n
l2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2\
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stack?
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2^
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stack_1?
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stack_2?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2StridedSliceR2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose:y:0c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stack:output:0e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stack_1:output:0e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2?
l2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOpqrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02n
l2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp?
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMulMatMul]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_2:output:0t2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2_
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul?
m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOprrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02o
m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp?
^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAddBiasAddg2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul:product:0u2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2`
^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd?
n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOpsrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02p
n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp?
_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1MatMulS2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros:output:0v2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2a
_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1?
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/addAddV2g2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd:output:0i2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2\
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/add?
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/TanhTanh^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2]
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/Tanh?
b2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2d
b2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2_1/element_shape?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2_1TensorListReservek2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2_1/element_shape:output:0]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2_1?
I2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2K
I2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/time?
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2_
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/maximum_iterations?
W2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2Y
W2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/loop_counter?
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/whileWhile`2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/loop_counter:output:0f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while/maximum_iterations:output:0R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/time:output:0]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2_1:handle:0S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/zeros:output:0]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_1:output:0|2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0qrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_readvariableop_resourcerrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_biasadd_readvariableop_resourcesrnn_reg_bidirect_swivel_balance_bidirectional_forward_simple_rnn_simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *b
bodyZRX
V2.3-RNN-Reg-Bidirect-Swivel-Balance_bidirectional_forward_simple_rnn_while_body_378136*b
condZRX
V2.3-RNN-Reg-Bidirect-Swivel-Balance_bidirectional_forward_simple_rnn_while_cond_378135*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2L
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while?
u2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2w
u2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
g2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStackS2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while:output:3~2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02i
g2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack?
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2\
Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stack?
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stack_1?
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2^
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stack_2?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3StridedSlicep2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stack:output:0e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stack_1:output:0e2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose_1/perm?
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose_1	Transposep2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2R
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/transpose_1?
K2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/ShapeShape<2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape:output:0*
T0*
_output_shapes
:2M
K2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/Shape?
Y2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2[
Y2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stack?
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2]
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stack_1?
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2]
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stack_2?
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_sliceStridedSliceT2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/Shape:output:0b2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stack:output:0d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stack_1:output:0d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2U
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice?
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2S
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/mul/y?
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/mulMul\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice:output:0Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2Q
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/mul?
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2T
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/Less/y?
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/LessLessS2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/mul:z:0[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2R
P2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/Less?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/packed/1?
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/packedPack\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice:output:0]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2T
R2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/packed?
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2S
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/Const?
K2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zerosFill[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/packed:output:0Z2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2M
K2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose/perm?
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose	Transpose<2.3-RNN-Reg-Bidirect-Swivel-Balance/reshape/Reshape:output:0]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2Q
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose?
M2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/Shape_1ShapeS2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose:y:0*
T0*
_output_shapes
:2O
M2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/Shape_1?
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2]
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stack?
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2_
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stack_1?
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stack_2?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1StridedSliceV2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/Shape_1:output:0d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stack:output:0f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stack_1:output:0f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1?
a2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2c
a2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2/element_shape?
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2TensorListReservej2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2/element_shape:output:0^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02U
S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2?
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2V
T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/ReverseV2/axis?
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/ReverseV2	ReverseV2S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose:y:0]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/ReverseV2/axis:output:0*
T0*+
_output_shapes
:?????????2Q
O2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/ReverseV2?
{2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2}
{2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorX2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/ReverseV2:output:0?2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02o
m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor?
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2]
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stack?
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2_
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stack_1?
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stack_2?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2StridedSliceS2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose:y:0d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stack:output:0f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stack_1:output:0f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2?
m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOprrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02o
m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp?
^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMulMatMul^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_2:output:0u2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2`
^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul?
n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOpsrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02p
n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAddBiasAddh2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul:product:0v2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2a
_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd?
o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOptrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02q
o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
`2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1MatMulT2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros:output:0w2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2b
`2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1?
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/addAddV2h2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd:output:0j2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2]
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/add?
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/TanhTanh_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2^
\2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/Tanh?
c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2e
c2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2_1/element_shape?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2_1TensorListReservel2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2_1/element_shape:output:0^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2_1?
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2L
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/time?
^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2`
^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/maximum_iterations?
X2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2Z
X2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/loop_counter?
K2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/whileWhilea2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/loop_counter:output:0g2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while/maximum_iterations:output:0S2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/time:output:0^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2_1:handle:0T2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/zeros:output:0^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_1:output:0}2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0rrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_readvariableop_resourcesrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_biasadd_readvariableop_resourcetrnn_reg_bidirect_swivel_balance_bidirectional_backward_simple_rnn_simple_rnn_cell_2_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *c
body[RY
W2.3-RNN-Reg-Bidirect-Swivel-Balance_bidirectional_backward_simple_rnn_while_body_378246*c
cond[RY
W2.3-RNN-Reg-Bidirect-Swivel-Balance_bidirectional_backward_simple_rnn_while_cond_378245*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2M
K2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while?
v2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2x
v2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape?
h2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStackTensorListStackT2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while:output:32.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????
*
element_dtype02j
h2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack?
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2]
[2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stack?
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2_
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stack_1?
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2_
]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stack_2?
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3StridedSliceq2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0d2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stack:output:0f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stack_1:output:0f2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2W
U2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3?
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2X
V2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose_1/perm?
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose_1	Transposeq2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/TensorArrayV2Stack/TensorListStack:tensor:0_2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????
2S
Q2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/transpose_1?
=2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2?
=2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/concat/axis?
82.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/concatConcatV2]2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/strided_slice_3:output:0^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/strided_slice_3:output:0F2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2:
82.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/concat?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul/ReadVariableOpReadVariableOpFrnn_reg_bidirect_swivel_balance_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul/ReadVariableOp?
22.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMulMatMulA2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/concat:output:0I2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
22.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul?
B2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd/ReadVariableOpReadVariableOpGrnn_reg_bidirect_swivel_balance_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
B2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd/ReadVariableOp?
32.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAddBiasAdd<2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul:product:0J2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????25
32.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd?
02.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/ReluRelu<2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
02.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/Relu?
62.3-RNN-Reg-Bidirect-Swivel-Balance/dropout_1/IdentityIdentity>2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/Relu:activations:0*
T0*'
_output_shapes
:?????????28
62.3-RNN-Reg-Bidirect-Swivel-Balance/dropout_1/Identity?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul/ReadVariableOpReadVariableOpFrnn_reg_bidirect_swivel_balance_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul/ReadVariableOp?
22.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMulMatMul?2.3-RNN-Reg-Bidirect-Swivel-Balance/dropout_1/Identity:output:0I2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
22.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul?
B2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd/ReadVariableOpReadVariableOpGrnn_reg_bidirect_swivel_balance_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
B2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd/ReadVariableOp?
32.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAddBiasAdd<2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul:product:0J2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????25
32.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd?
02.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/ReluRelu<2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
02.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/Relu?
12.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@23
12.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/Const?
32.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    25
32.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/Const_1?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/clip_by_value/MinimumMinimum>2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/Relu:activations:0:2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/Const:output:0*
T0*'
_output_shapes
:?????????2C
A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/clip_by_value/Minimum?
92.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/clip_by_valueMaximumE2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/clip_by_value/Minimum:z:0<2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/Const_1:output:0*
T0*'
_output_shapes
:?????????2;
92.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/clip_by_value?
IdentityIdentity=2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/clip_by_value:z:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?	
NoOpNoOpo^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpn^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpp^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpL^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/whilen^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpm^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpo^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpK^2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/whileC^2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd/ReadVariableOpB^2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul/ReadVariableOpC^2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd/ReadVariableOpB^2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul/ReadVariableOpB^2.3-RNN-Reg-Bidirect-Swivel-Balance/embed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : : : 2?
n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOpn2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/BiasAdd/ReadVariableOp2?
m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOpm2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul/ReadVariableOp2?
o2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOpo2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/simple_rnn_cell_2/MatMul_1/ReadVariableOp2?
K2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/whileK2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/backward_simple_rnn/while2?
m2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOpm2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/BiasAdd/ReadVariableOp2?
l2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOpl2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul/ReadVariableOp2?
n2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOpn2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/simple_rnn_cell_1/MatMul_1/ReadVariableOp2?
J2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/whileJ2.3-RNN-Reg-Bidirect-Swivel-Balance/bidirectional/forward_simple_rnn/while2?
B2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd/ReadVariableOpB2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/BiasAdd/ReadVariableOp2?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul/ReadVariableOpA2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_2/MatMul/ReadVariableOp2?
B2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd/ReadVariableOpB2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/BiasAdd/ReadVariableOp2?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul/ReadVariableOpA2.3-RNN-Reg-Bidirect-Swivel-Balance/dense_3/MatMul/ReadVariableOp2?
A2.3-RNN-Reg-Bidirect-Swivel-Balance/embed/StatefulPartitionedCallA2.3-RNN-Reg-Bidirect-Swivel-Balance/embed/StatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_nameInput:

_output_shapes
: 
?#
?
while_body_378398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_1_378420_0:
.
 while_simple_rnn_cell_1_378422_0:
2
 while_simple_rnn_cell_1_378424_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_1_378420:
,
while_simple_rnn_cell_1_378422:
0
while_simple_rnn_cell_1_378424:

??/while/simple_rnn_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
/while/simple_rnn_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_1_378420_0 while_simple_rnn_cell_1_378422_0 while_simple_rnn_cell_1_378424_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_37838521
/while/simple_rnn_cell_1/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity8while/simple_rnn_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp0^while/simple_rnn_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_1_378420 while_simple_rnn_cell_1_378420_0"B
while_simple_rnn_cell_1_378422 while_simple_rnn_cell_1_378422_0"B
while_simple_rnn_cell_1_378424 while_simple_rnn_cell_1_378424_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2b
/while/simple_rnn_cell_1/StatefulPartitionedCall/while/simple_rnn_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_embed_layer_call_fn_381587

inputs
unknown
	unknown_0	
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_embed_layer_call_and_return_conditional_losses_3800732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?I
?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_379935

inputsB
0simple_rnn_cell_1_matmul_readvariableop_resource:
?
1simple_rnn_cell_1_biasadd_readvariableop_resource:
D
2simple_rnn_cell_1_matmul_1_readvariableop_resource:


identity??(simple_rnn_cell_1/BiasAdd/ReadVariableOp?'simple_rnn_cell_1/MatMul/ReadVariableOp?)simple_rnn_cell_1/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????????????*
shrink_axis_mask2
strided_slice_2?
'simple_rnn_cell_1/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02)
'simple_rnn_cell_1/MatMul/ReadVariableOp?
simple_rnn_cell_1/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul?
(simple_rnn_cell_1/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(simple_rnn_cell_1/BiasAdd/ReadVariableOp?
simple_rnn_cell_1/BiasAddBiasAdd"simple_rnn_cell_1/MatMul:product:00simple_rnn_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/BiasAdd?
)simple_rnn_cell_1/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02+
)simple_rnn_cell_1/MatMul_1/ReadVariableOp?
simple_rnn_cell_1/MatMul_1MatMulzeros:output:01simple_rnn_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/MatMul_1?
simple_rnn_cell_1/addAddV2"simple_rnn_cell_1/BiasAdd:output:0$simple_rnn_cell_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/add?
simple_rnn_cell_1/TanhTanhsimple_rnn_cell_1/add:z:0*
T0*'
_output_shapes
:?????????
2
simple_rnn_cell_1/Tanh?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_1_matmul_readvariableop_resource1simple_rnn_cell_1_biasadd_readvariableop_resource2simple_rnn_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_379869*
condR
while_cond_379868*8
output_shapes'
%: : : : :?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1s
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp)^simple_rnn_cell_1/BiasAdd/ReadVariableOp(^simple_rnn_cell_1/MatMul/ReadVariableOp*^simple_rnn_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'???????????????????????????: : : 2T
(simple_rnn_cell_1/BiasAdd/ReadVariableOp(simple_rnn_cell_1/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_1/MatMul/ReadVariableOp'simple_rnn_cell_1/MatMul/ReadVariableOp2V
)simple_rnn_cell_1/MatMul_1/ReadVariableOp)simple_rnn_cell_1/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
%backward_simple_rnn_while_cond_382533D
@backward_simple_rnn_while_backward_simple_rnn_while_loop_counterJ
Fbackward_simple_rnn_while_backward_simple_rnn_while_maximum_iterations)
%backward_simple_rnn_while_placeholder+
'backward_simple_rnn_while_placeholder_1+
'backward_simple_rnn_while_placeholder_2F
Bbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382533___redundant_placeholder0\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382533___redundant_placeholder1\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382533___redundant_placeholder2\
Xbackward_simple_rnn_while_backward_simple_rnn_while_cond_382533___redundant_placeholder3&
"backward_simple_rnn_while_identity
?
backward_simple_rnn/while/LessLess%backward_simple_rnn_while_placeholderBbackward_simple_rnn_while_less_backward_simple_rnn_strided_slice_1*
T0*
_output_shapes
: 2 
backward_simple_rnn/while/Less?
"backward_simple_rnn/while/IdentityIdentity"backward_simple_rnn/while/Less:z:0*
T0
*
_output_shapes
: 2$
"backward_simple_rnn/while/Identity"Q
"backward_simple_rnn_while_identity+backward_simple_rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
:
?
?
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_378505

inputs

states0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

MatMul_1k
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
addO
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????
2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityg

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity_1?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_namestates
?1
?
while_body_383485
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_2_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_2_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_2_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:

??.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_2/MatMul/ReadVariableOp?/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????????29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:??????????????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
-while/simple_rnn_cell_2/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype02/
-while/simple_rnn_cell_2/MatMul/ReadVariableOp?
while/simple_rnn_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
while/simple_rnn_cell_2/MatMul?
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype020
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp?
while/simple_rnn_cell_2/BiasAddBiasAdd(while/simple_rnn_cell_2/MatMul:product:06while/simple_rnn_cell_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2!
while/simple_rnn_cell_2/BiasAdd?
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype021
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp?
 while/simple_rnn_cell_2/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2"
 while/simple_rnn_cell_2/MatMul_1?
while/simple_rnn_cell_2/addAddV2(while/simple_rnn_cell_2/BiasAdd:output:0*while/simple_rnn_cell_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/add?
while/simple_rnn_cell_2/TanhTanhwhile/simple_rnn_cell_2/add:z:0*
T0*'
_output_shapes
:?????????
2
while/simple_rnn_cell_2/Tanh?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_2/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity while/simple_rnn_cell_2/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????
2
while/Identity_4?

while/NoOpNoOp/^while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_2/MatMul/ReadVariableOp0^while/simple_rnn_cell_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_2_biasadd_readvariableop_resource9while_simple_rnn_cell_2_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_2_matmul_1_readvariableop_resource:while_simple_rnn_cell_2_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_2_matmul_readvariableop_resource8while_simple_rnn_cell_2_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????
: : : : : 2`
.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp.while/simple_rnn_cell_2/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_2/MatMul/ReadVariableOp-while/simple_rnn_cell_2/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp/while/simple_rnn_cell_2/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????
:

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_embed_layer_call_and_return_conditional_losses_381609

inputs
unknown
	unknown_0	
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_restored_function_body_2217052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_3:0StatefulPartitionedCall_48"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
3
Input*
serving_default_Input:0?????????=
dense_32
StatefulPartitionedCall_2:0?????????tensorflow/serving/predict2(

asset_path_initializer:0
tokens.txt:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
	_func
	_callable

Variable:0
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
forward_layer
backward_layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
-iter

.beta_1

/beta_2
	0decay
1learning_ratem?m?m?'m?(m?2m?3m?4m?5m?6m?7m?v?v?v?'v?(v?2v?3v?4v?5v?6v?7v?"
	optimizer
n
0
21
32
43
54
65
76
7
8
'9
(10"
trackable_list_wrapper
n
0
21
32
43
54
65
76
7
8
'9
(10"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
8layer_regularization_losses
9layer_metrics
	trainable_variables

regularization_losses

:layers
;metrics
<non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
=_vocabulary_file

>_table

embeddings
?	variables
?trainable_variables
@
signatures
#A_self_saveable_object_factories
?__call__
?_preprocess"
_generic_user_object
:
??2Variable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Blayer_regularization_losses
Clayer_metrics
trainable_variables
regularization_losses

Dlayers
Emetrics
Fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Glayer_regularization_losses
Hlayer_metrics
trainable_variables
regularization_losses

Ilayers
Jmetrics
Knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Lcell
M
state_spec
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
Rcell
S
state_spec
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
J
20
31
42
53
64
75"
trackable_list_wrapper
J
20
31
42
53
64
75"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Xlayer_regularization_losses
Ylayer_metrics
trainable_variables
regularization_losses

Zlayers
[metrics
\non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
]layer_regularization_losses
^layer_metrics
 trainable_variables
!regularization_losses

_layers
`metrics
anon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#	variables
blayer_regularization_losses
clayer_metrics
$trainable_variables
%regularization_losses

dlayers
emetrics
fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)	variables
glayer_regularization_losses
hlayer_metrics
*trainable_variables
+regularization_losses

ilayers
jmetrics
knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
K:I
29bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel
U:S

2Cbidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel
E:C
27bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias
L:J
2:bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel
V:T

2Dbidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel
F:D
28bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
 "
trackable_list_wrapper
* 
C
?_create_resource
?_initialize
?_destroy_resourceR 
'
0"
trackable_list_wrapper
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

2kernel
3recurrent_kernel
4bias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
?
N	variables
slayer_regularization_losses
tlayer_metrics
Otrainable_variables

ustates
Pregularization_losses

vlayers
wmetrics
xnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

5kernel
6recurrent_kernel
7bias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
 "
trackable_list_wrapper
?
T	variables
}layer_regularization_losses
~layer_metrics
Utrainable_variables

states
Vregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
?
o	variables
 ?layer_regularization_losses
?layer_metrics
ptrainable_variables
qregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
5
50
61
72"
trackable_list_wrapper
 "
trackable_list_wrapper
?
y	variables
 ?layer_regularization_losses
?layer_metrics
ztrainable_variables
{regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
!:
??2Adam/Variable/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
P:N
2@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/m
Z:X

2JAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/m
J:H
2>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/m
Q:O
2AAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/m
[:Y

2KAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/m
K:I
2?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/m
!:
??2Adam/Variable/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:#2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
P:N
2@Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/kernel/v
Z:X

2JAdam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/recurrent_kernel/v
J:H
2>Adam/bidirectional/forward_simple_rnn/simple_rnn_cell_1/bias/v
Q:O
2AAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/kernel/v
[:Y

2KAdam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/recurrent_kernel/v
K:I
2?Adam/bidirectional/backward_simple_rnn/simple_rnn_cell_2/bias/v
?2?
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_380408
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_381020
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_381051
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_380878?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_378333Input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_381310
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_381576
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_380914
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_380950?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_embed_layer_call_fn_381587
&__inference_embed_layer_call_fn_381598?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_embed_layer_call_and_return_conditional_losses_381609
A__inference_embed_layer_call_and_return_conditional_losses_381620?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_reshape_layer_call_fn_381625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_reshape_layer_call_and_return_conditional_losses_381638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_bidirectional_layer_call_fn_381655
.__inference_bidirectional_layer_call_fn_381672
.__inference_bidirectional_layer_call_fn_381689
.__inference_bidirectional_layer_call_fn_381706?
???
FullArgSpecO
argsG?D
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults?
p 

 

 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_bidirectional_layer_call_and_return_conditional_losses_381930
I__inference_bidirectional_layer_call_and_return_conditional_losses_382154
I__inference_bidirectional_layer_call_and_return_conditional_losses_382378
I__inference_bidirectional_layer_call_and_return_conditional_losses_382602?
???
FullArgSpecO
argsG?D
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults?
p 

 

 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_2_layer_call_fn_382611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_382622?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_1_layer_call_fn_382627
*__inference_dropout_1_layer_call_fn_382632?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_382637
E__inference_dropout_1_layer_call_and_return_conditional_losses_382649?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_3_layer_call_fn_382658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_3_layer_call_and_return_conditional_losses_382673?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_380989Input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference___call___115?
???
FullArgSpec
args?
j	sentences
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
??????????
?2?
__inference__preprocess_92?
???
FullArgSpec
args?
j	sentences
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
??????????
?2?
3__inference_forward_simple_rnn_layer_call_fn_382684
3__inference_forward_simple_rnn_layer_call_fn_382695
3__inference_forward_simple_rnn_layer_call_fn_382706
3__inference_forward_simple_rnn_layer_call_fn_382717?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_382829
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_382941
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_383053
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_383165?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_backward_simple_rnn_layer_call_fn_383176
4__inference_backward_simple_rnn_layer_call_fn_383187
4__inference_backward_simple_rnn_layer_call_fn_383198
4__inference_backward_simple_rnn_layer_call_fn_383209?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383323
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383437
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383551
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383665?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__creator_128?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_134?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2??
???
FullArgSpec
args?
jself
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2B0
!__inference_signature_wrapper_123	sentences
?2?
2__inference_simple_rnn_cell_1_layer_call_fn_383679
2__inference_simple_rnn_cell_1_layer_call_fn_383693?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_383710
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_383727?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_simple_rnn_cell_2_layer_call_fn_383741
2__inference_simple_rnn_cell_2_layer_call_fn_383755?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_383772
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_383789?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
	J
Const?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_380914k>?243576'(2?/
(?%
?
Input?????????
p 

 
? "%?"
?
0?????????
? ?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_380950k>?243576'(2?/
(?%
?
Input?????????
p

 
? "%?"
?
0?????????
? ?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_381310l>?243576'(3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
___inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_and_return_conditional_losses_381576l>?243576'(3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_380408^>?243576'(2?/
(?%
?
Input?????????
p 

 
? "???????????
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_380878^>?243576'(2?/
(?%
?
Input?????????
p

 
? "???????????
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_381020_>?243576'(3?0
)?&
?
inputs?????????
p 

 
? "???????????
D__inference_2.3-RNN-Reg-Bidirect-Swivel-Balance_layer_call_fn_381051_>?243576'(3?0
)?&
?
inputs?????????
p

 
? "??????????l
__inference___call___115P>?.?+
$?!
?
	sentences?????????
? "??????????4
__inference__creator_128?

? 
? "? <
__inference__initializer_134=>?

? 
? "? ?
__inference__preprocess_92?>?.?+
$?!
?
	sentences?????????
? "I?F
?
0?????????	
?
1?????????	
?
2	?
!__inference__wrapped_model_378333o>?243576'(*?'
 ?
?
Input?????????
? "1?.
,
dense_3!?
dense_3??????????
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383323}576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????

? ?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383437}576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????

? ?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383551576Q?N
G?D
6?3
inputs'???????????????????????????

 
p 

 
? "%?"
?
0?????????

? ?
O__inference_backward_simple_rnn_layer_call_and_return_conditional_losses_383665576Q?N
G?D
6?3
inputs'???????????????????????????

 
p

 
? "%?"
?
0?????????

? ?
4__inference_backward_simple_rnn_layer_call_fn_383176p576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????
?
4__inference_backward_simple_rnn_layer_call_fn_383187p576O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????
?
4__inference_backward_simple_rnn_layer_call_fn_383198r576Q?N
G?D
6?3
inputs'???????????????????????????

 
p 

 
? "??????????
?
4__inference_backward_simple_rnn_layer_call_fn_383209r576Q?N
G?D
6?3
inputs'???????????????????????????

 
p

 
? "??????????
?
I__inference_bidirectional_layer_call_and_return_conditional_losses_381930?243576\?Y
R?O
=?:
8?5
inputs/0'???????????????????????????
p 

 

 

 
? "%?"
?
0?????????
? ?
I__inference_bidirectional_layer_call_and_return_conditional_losses_382154?243576\?Y
R?O
=?:
8?5
inputs/0'???????????????????????????
p

 

 

 
? "%?"
?
0?????????
? ?
I__inference_bidirectional_layer_call_and_return_conditional_losses_382378t243576C?@
9?6
$?!
inputs?????????
p 

 

 

 
? "%?"
?
0?????????
? ?
I__inference_bidirectional_layer_call_and_return_conditional_losses_382602t243576C?@
9?6
$?!
inputs?????????
p

 

 

 
? "%?"
?
0?????????
? ?
.__inference_bidirectional_layer_call_fn_381655?243576\?Y
R?O
=?:
8?5
inputs/0'???????????????????????????
p 

 

 

 
? "???????????
.__inference_bidirectional_layer_call_fn_381672?243576\?Y
R?O
=?:
8?5
inputs/0'???????????????????????????
p

 

 

 
? "???????????
.__inference_bidirectional_layer_call_fn_381689g243576C?@
9?6
$?!
inputs?????????
p 

 

 

 
? "???????????
.__inference_bidirectional_layer_call_fn_381706g243576C?@
9?6
$?!
inputs?????????
p

 

 

 
? "???????????
C__inference_dense_2_layer_call_and_return_conditional_losses_382622\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_2_layer_call_fn_382611O/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_3_layer_call_and_return_conditional_losses_382673\'(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_3_layer_call_fn_382658O'(/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_382637\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_382649\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_1_layer_call_fn_382627O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_1_layer_call_fn_382632O3?0
)?&
 ?
inputs?????????
p
? "???????????
A__inference_embed_layer_call_and_return_conditional_losses_381609^>?/?,
%?"
?
inputs?????????
p 
? "%?"
?
0?????????
? ?
A__inference_embed_layer_call_and_return_conditional_losses_381620^>?/?,
%?"
?
inputs?????????
p
? "%?"
?
0?????????
? {
&__inference_embed_layer_call_fn_381587Q>?/?,
%?"
?
inputs?????????
p 
? "??????????{
&__inference_embed_layer_call_fn_381598Q>?/?,
%?"
?
inputs?????????
p
? "???????????
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_382829}243O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????

? ?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_382941}243O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????

? ?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_383053243Q?N
G?D
6?3
inputs'???????????????????????????

 
p 

 
? "%?"
?
0?????????

? ?
N__inference_forward_simple_rnn_layer_call_and_return_conditional_losses_383165243Q?N
G?D
6?3
inputs'???????????????????????????

 
p

 
? "%?"
?
0?????????

? ?
3__inference_forward_simple_rnn_layer_call_fn_382684p243O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????
?
3__inference_forward_simple_rnn_layer_call_fn_382695p243O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????
?
3__inference_forward_simple_rnn_layer_call_fn_382706r243Q?N
G?D
6?3
inputs'???????????????????????????

 
p 

 
? "??????????
?
3__inference_forward_simple_rnn_layer_call_fn_382717r243Q?N
G?D
6?3
inputs'???????????????????????????

 
p

 
? "??????????
?
C__inference_reshape_layer_call_and_return_conditional_losses_381638\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? {
(__inference_reshape_layer_call_fn_381625O/?,
%?"
 ?
inputs?????????
? "???????????
!__inference_signature_wrapper_123x>?;?8
? 
1?.
,
	sentences?
	sentences?????????"3?0
.
output_0"?
output_0??????????
$__inference_signature_wrapper_380989x>?243576'(3?0
? 
)?&
$
Input?
Input?????????"1?.
,
dense_3!?
dense_3??????????
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_383710?243\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????

p 
? "R?O
H?E
?
0/0?????????

$?!
?
0/1/0?????????

? ?
M__inference_simple_rnn_cell_1_layer_call_and_return_conditional_losses_383727?243\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????

p
? "R?O
H?E
?
0/0?????????

$?!
?
0/1/0?????????

? ?
2__inference_simple_rnn_cell_1_layer_call_fn_383679?243\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????

p 
? "D?A
?
0?????????

"?
?
1/0?????????
?
2__inference_simple_rnn_cell_1_layer_call_fn_383693?243\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????

p
? "D?A
?
0?????????

"?
?
1/0?????????
?
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_383772?576\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????

p 
? "R?O
H?E
?
0/0?????????

$?!
?
0/1/0?????????

? ?
M__inference_simple_rnn_cell_2_layer_call_and_return_conditional_losses_383789?576\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????

p
? "R?O
H?E
?
0/0?????????

$?!
?
0/1/0?????????

? ?
2__inference_simple_rnn_cell_2_layer_call_fn_383741?576\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????

p 
? "D?A
?
0?????????

"?
?
1/0?????????
?
2__inference_simple_rnn_cell_2_layer_call_fn_383755?576\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????

p
? "D?A
?
0?????????

"?
?
1/0?????????
