??%
?,?,
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
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
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
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
A
SelectV2
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
executor_typestring ??
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
Ttype"
Indextype:
2	"

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
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??#
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ȩd*&
shared_nameAdam/dense_4/kernel/v
?
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
Ȩd*
dtype0
?
Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv1d_2/bias/v
z
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv1d_2/kernel/v
?
*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*$
_output_shapes
:??*
dtype0
?
Adam/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*.
shared_nameAdam/embedding_2/embeddings/v
?
1Adam/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/v*!
_output_shapes
:???*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ȩd*&
shared_nameAdam/dense_4/kernel/m
?
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
Ȩd*
dtype0
?
Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv1d_2/bias/m
z
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv1d_2/kernel/m
?
*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*$
_output_shapes
:??*
dtype0
?
Adam/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*.
shared_nameAdam/embedding_2/embeddings/m
?
1Adam/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/m*!
_output_shapes
:???*
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
 *0
config_proto 

CPU

GPU2*0J 8? *4
f/R-
+__inference_restored_function_body_11792225
?
StatefulPartitionedCall_1StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *4
f/R-
+__inference_restored_function_body_11792230
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
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:d*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:d*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ȩd*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
Ȩd*
dtype0
s
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_2/bias
l
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes	
:?*
dtype0
?
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv1d_2/kernel
y
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*$
_output_shapes
:??*
dtype0
?
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*'
shared_nameembedding_2/embeddings
?
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*!
_output_shapes
:???*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes

:??*
dtype0*??
value??B????BtheBiBtoBaByouBandBisBthatBitBnameBofBthisBforBinBmyBwasBbutBnotBbeBsoBjustBonBhaveBlikeBareBwithBmeByourBitsBwhatBtheyBheBifBimBasBatBloveBaboutBallBnoBgoodBdoBgetBoneBweBdontBwouldBhowBpeopleBcanBoutBreallyBupBanBknowBthinkBtooBwhenBfromBorBmoreBthanksBthatsBthereBthemBsheBherBwillBnowBlolBhimBhisBtimeBseeBsomeBthankBhasBbecauseBi’mBwhyBmuchBevenBohBwhoBhadBwellBstillBbeenBgotBrightBhereBonlyBbadBtheirBit’sBneverBgoBbeingBthanBbyBwantBdidBsorryByeahByoureBgreatBhopeBmakeBveryBwereBwayBfeelBthenBmanBgoingBbetterBshouldBthingBalsoBprettyBactuallyBdon’tBbackBsameBcantBcouldBsomeoneBsayBsureBanyBamBdidntBbestBneedBgameBhappyBthoughBthoseB	somethingBotherBdayBiveBthat’sBthoughtBwowBlookBmakesBtheseBalwaysBlooksByearBgladBusBoffBmadeBguyBintoByesBtakeBfirstBoverBmostBourBworkBafterBeverBlifeByearsBhateBdamnBnewBmeanBdoesntBhesBpostBniceBbeforeBdoesBthingsBmaybeBgettingBprobablyBsaidBwhereBdownBfuckBfindBlastBwrongBhardBlotBwishBhelpBkeepBdoingBanythingBshitBfunByou’reBeveryoneBfuckingBeveryBdudeBoldBmanyBidBtryBpointBlittleBhahaBcoolBpersonBgiveBsoundsBrealBisntBagainBreadBstopBweirdBillBlongBplayBleastBawesomeBguysBbelieveBnothingBsuchBamazingBseenBuseBokBhavingBtellBenoughBpleaseBbothBcommentBtryingBluckBcomeBsubBaroundBanyoneBfunnyBkindBmightBalreadyBwhileBwhichBgonnaB2BguessBseemsBtrueBinterestingBhellBsadB
definitelyBwaitB3BideaBletBstupidBbigBtwoBcan’tBpartBhearBtheyreBagreeBdidn’tByourselfBsinceB
understandBnextBanotherBteamBshowBputBwatchBwithoutBthroughBlookingBdoneBproblemBliveBjobBsayingBi’veBkidsBelseBuntilB	literallyBawayBfriendBgetsBhonestlyBfavoriteBrememberBvideoBmoneyBendBusedBcareBworldBlmaoBtimesB
everythingBfarBfriendsBexactlyBworstBownBenjoyBredditBreasonBfewBworseBplaceBmustBonceBbitBshesBtheresBjokeBheardBeitherBterribleBhe’sBcallBleftBagainstBmyselfBhappenedBmakingBwholeBseasonBstuffBgirlByetBomgB	differentBwentBagoBstartBsawBalmostB	beautifulBpoorBwomenBstoryBokayBmayBcrazyBsuperBtalkingBkidBlessBahB
absolutelyBopBhighBchangeBtodayBmindBschoolBheyBtalkBfaceBimagineB	surprisedBplayingBwouldntBsenseBhomeBwinBthinkingBwasntBneedsBfactBfeelingBfoundB
appreciateBsecondBstayBwantedBfamilyBcameBcuteBfreeBdaysBgamesBwonderBgottaBwontB	sometimesBawfulBwatchingBlostBkindaB1B5BtakingBnewsBhappenBcomingBseemB	doesn’tBheadBfeelsBweekBfanBwelcomeBmissBquestionBsucksBgodBleaveBwhatsBuBmenB	seriouslyBtotallyBableBtookBsexBarentBwomanBmatterBhorribleBlovedBrealizeBonesBmoveBtopBletsBaskBi’dBi’llBcommentsBfineBdogBdealBthoBsBeditBrunBlaughBeatB
completelyBbroBhitB
especiallyBmovieBsoonBseeingBcheckBnightBkillBgoesBmomBtriedBbetweenBworthBcauseBaloneBsupportBpayBinsteadBwordsBknewBquiteBloseBdumbBusingB	they’reBsoundBmeantBtogetherBtoldBratherBplayersBcarBhurtBheartBwantsBreadingBentireBcalledBbabyBcaseBadviceBthreadBreligionBworryBhappensBcuriousBanymoreBshameByouveBsideBmeansBisn’tBwordBdieBhalfBchildBsweetBfullBfoodBhaventBcloseBhotBcomesBexcitedBrestBeachB4BparentsBdeadByaBtbhBopinionBplayedBfutureBforgotBworksBholyBeasyBmineBcountryBbuyBboyB10BstrongBsaysB
disgustingBbecomeB	situationB	hopefullyBfinallyBshotBpictureBstartedBshe’sBperfectBcakeBnahBpostedBgoneBcoupleB20BunfortunatelyBinternetBothersBopenBpickBfairBchanceBcongratsBexceptBsingleBpossibleBlookedB100B😂BsharingBfakeB
experienceBunderBmomentByoungBstateBmissedBissueBmonthsBdBanywayBworkingBdeserveBnobodyBclearlyBbreakBpastBsickBgirlsBepisodeBhugeBfansBgaveBdadBmadBidkBlivingBseriousBhopingBlovesBwifeBcourseBscaredBlearnBwannaB	obviouslyB	hilariousByoullBlikedBanswerB6BresponseBpostsBneededBmemeBhairBarticleB	there’sBgivingBlineBhouseBfeltBuglyB
governmentBrelationshipBredBageB	characterBidiotBlinkBchildrenBworriedBracistBlateBphoneBlaterBcouldntBafraidBtakesBmateBbodyBfightBlevelBpainBforgetBcoldBactualBusuallyBoftenBlossBbringBhistoryByepBnormalBconfusedBturnBproudBminutesB	subredditBsongBhoursBcatB
ridiculousBsafeBcheersBdeathBbrotherBwatchedBgivenB	basicallyBassB
personallyBhumanBbetBdateBoutsideBtypeBnoticedBlikelyBwhateverBsmallBreplyBlowBgunBtrulyBmusicBbeatBwaterBeyesBughBplayerBluckyBknowsB
interestedBaccountBtonightB	importantBangryBwasn’tBcryingBunlessBstraightBkilledBduringBtitleBhandBcringeBpostingBpowerB
themselvesBargumentBnumberBhahahaBcorrectBcityB	wonderfulBhaBtrustBroomB	extremelyBdoubtBpassB	wonderingBweeksBmonthBcallingBcutByeaBupvoteBshutBhandsBissuesBexpectBwaitingBthreeB	attentionBcreepyByoutubeBshowsBmotherBsaveBsadlyBgroupB30BexplainBselfBblackBgoldBwe’reBsuckBaskingBgivesB
apparentlyBpartyBamericaBpointsBfastBbookB
wouldn’tBwhiteBinsaneB
differenceBdiedBareaBnopeBhourBalrightBhusbandBhuhBcontrolBblameBlikesB	downvotedBwhat’sBvoteBsonBlistenBforwardBtellingBsimilarBsortBpoliceBoriginalBbirthdayBbehindBlosingBtruthBsocialBbrainB8BstandBshortBamountB15BtradeBdriveBsourceBcryBhugBsignBconsideringBcongratulationsB	fantasticBsmileBmentalBcrapBbuddyBthinksBreadyBleagueBmorningBclearBfigureBclassBwonBearlyBupsetBsystemBsurpriseBproblemsBcollegeBhealthBannoyingBinfoBgottenBanxietyBhimselfBballBlawB	genuinelyBtrashBfireBcommonBrunningBlongerBhonestBfallB	communityBshareBscaryBfeelingsBchoiceBsetBoddBmistakeBimoBsocietyBmediaBinsideB	yesterdayBeatingBcompanyBbasedB7BlotsBaddBtakenB12BremindsBdueBtrollBmostlyBlistBwasteBspaceBawkwardBworkedBmiddleBrespectBhelpsBstartingBshouldntBpissedB	otherwiseBexampleBendedBbannedBallowedBaliveBhurtsBchangedBaskedBusernameBpublicBkillingB
incrediblyBfrontBdreamBboysBplanBdarkBholdBgarbageBblueByoudBabsoluteBlaughingBgoalBbedBwearBstoreBsleepBlieBwon’tBspendBfatBavoidBtvBteamsBspecialB	happeningBexactBthoughtsBhelpedBdrinkBbiggestByupBvideosBtiredBlaughedBaren’tBrealityBidiotsBbunchBactBsceneBregretBinformationBfellowBeyeBcaresBbusinessB
couldn’tBbottomBtoxicBthrowBstickBseriesB	haven’tBfaultBexistBspeakBsimpleBrealizedBquickBorderBonlineBpayingBfootballBtwitterBtermBsomehowB	expectingBdisagreeB
attractiveBtoughBnoticeBeffortBeasierBbullshitBstepBmissingBbtwBbrokenBrichB	politicalBpieceBpersonalBolderBdoubleBafBphotoBpaidBobviousBlearnedBgrowBgeneralBevidenceBagreedBsexualBserviceBfearB	favouriteBcompleteBturnedBtomorrowBeasilyBdownvoteBdoctorBwallB	somewhereBnearBjealousBdareB	dangerousBhatedBblessBmemesBlovingBcareerBcannotBboringBsendBdisappointedBtasteBputtingBmessageBillegalBforeverBdefenseBvsBplacesBoptionBofficeBkeptBetcBehBdecentBstrangeBextraBenjoyedBendingBbloodBalthoughBwalkBpositiveBmouthBmessBexcuseBdropBdeepByou’veBpornBpoliticsBmetBitllBplentyBpicBignorantBgrossBstoriesBcanadaBabuseBwriteBtroubleBtheyllBqualityBknowingBhiB	certainlyBartByou’llByallBroundBquitBpainfulB	questionsBcleanBbarelyBtownBpullBinterestBpBhearingBremovedBloudBlegalBkarmaBattackBsupposedB	referenceBpositionBmainBlovelyBconversationBstuckBsimplyBshockedBrapeBpreferBminuteBiceBbroughtBtwiceBnoneBmatchBhelpfulBfollowBcatchBxdBstarBskinBpizzaBmarriedBlivesBkeepsBfuckedB	everybodyBclaimBbiggerBwearingBlightB	difficultB
charactersBassumeBfaithBexpectedBdatingBcomparedBbrokeBbitchBalongBuncomfortableBrecentlyB	presidentB
incredibleBdrunkBversionBfitBcopsBanimalsBaboveBspentBrandomBdyingBdogsByoBpriceBhasntBcontextBbotByikesBwarBthirdBlackBfatherBfalseBexBtextBmultipleBfightingBdrugsBdrivingBdeservesBaintB2ndBwithinBraceBawareBwtfBtillBsentB
depressionBcurrentBboughtBaverageBaccurateBvoiceBsmellBsirBreactionBmovingBjokesBfocusB	expensiveBcaughtBstrangerBspotBplusBfactsBcardByoursBsisterBladyBhugsBhealthyBgunsBblockBweedBseemedBrulesBpageBmedicalBhoweverBgoogleBanywaysBrecordBlegitBfixBemBdraftBcontinueBconsiderBbeerB2019BzeroBsmartBsecondsBheroBdecidedBburnB40BsitBevilBballsBrelatedBmurderBharderBcontentBweveBwerentBsouthBroadBmentionBchooseBbudBstoppedBreportBcatsB❤️BvalueBscienceB
eventuallyBwinningBperBmovedBlocalBkeepingBhighlyBhangBenglishBbornBbadlyBsavedBrightsBmeetBleadBfacebookB
consideredBcalmBadorableBwillingBsubsBsmhBruleBriskBquicklyBnastyBfindingBacceptBviewBtaxB	concernedB	boyfriendBappB25BtearsBsummerBpeaceBoBmoviesBlinesBignoreBbehaviorBwalkingBstatesBlogicBleavingB	excellentBtestB	statementBsentenceBlanguageBindeedB	availableBwaysB
terrifyingBstealBsomebodyBknownBjobsBcostBadmitBadBweightBwarmBhumansBgroundB
girlfriendB	deliciousBturnsBteacherBsexyBreasonsB	perfectlyBnonBnaturalBdespiteBwakeBupdateBtipBrudeBpossiblyBembarrassingBassumingBacrossBtripBqueenBnegativeBmillionBhatesB	happinessBforceBdailyBcontractB
constantlyBcertainB	brilliantBtherapyBsakeBfrustratingBcancerBamericanBuhBslightlyBfinishBwildBweekendBunderstandingBstyleBsillyBplaysBpartnerBendsBdamageBbuildBassholeBwinsBsiteBitselfBinjuryB	generallyBformBfinalBearthBearlierBdecideBclubBclassicBactionBsellBprotectBhandleBfreakingB	currentlyBcountBcontactBcheatingBadultB9BridBplayoffsBcoachB	christmasBanywhereBsittingBscreenBofferBflairBenjoyingBdaughterBcomplainBcarefulBworriesBoppositeBmrB	mentionedBmarriageBfellBdependsB“iBtrollingBtrainBsuddenlyBripBproofBkiddingBfeetBentirelyBdoorBthrowingBsolidBshirtBrockBrareBparentBlearningBlaB	countriesBcheapBwrittenBtowardsBperiodBukBshootBshittyB	recommendB
impossibleBgiftBfixedBdefendBdecisionBcreditBbreakingB	apologiesBanybodyBquoteBpissBlordBfailedByoungerB	wholesomeBtrafficBspecificBrideB	listeningBjumpBcheckedBbowlBwebsiteBshowingBroughBoilBnearlyBgreenBdearBwroteBwhosBwheneverBwestBpmBpicturesBnorthBnamesBlet’sBhigherBhelpingB	emotionalBdumbassBcolorBchillBwouldveBswearBprisonBliesBjerkBfolksBcheeseBchannelBbesidesBtheyveBsportsBspeakingBsendingBruinedBopinionsBnBmodeBmassiveBfiguredBchurchBbarBalbumBxBtouchBracismBneitherBmakeupBlivedBlegsBfloorBfiveBenergyBdeleteBcoffeeBbusyBargueB	apologizeBairByayBshootingBpurposeBphysicalBpatheticBlawsBinvolvedBboardBattitudeBtermsBsurelyBpopularBpenisBmoronB	interviewBdrugB	beginningBtotalBpureBjoinB
backgroundBaheadBaddedBurBruinBprayBlettingBcrimeBchickenBwritingBsuccessBsoulBjailBinnocentBfinishedBentertainingBcoverB
confidenceBarmsBapartBadultsBvisitBslowBshoesBreturnBresearchBoffenseBmemoryB	knowledgeBideasBfiredBexistedB
discussionBconcernB0B	insuranceBimageBholdingB
everywhereBcultureBcausedBcarsBbrutalBattemptB50BrelateBoofBmomentsBmobileBeffectBcuzBcourtBbooksBaffordBaccidentallyB1stB13BvotedB	triggeredBsnowBsearchB
reasonableBprofessionalBpretendBmeaningBmarketBheresBgreatestBfailBusaB
regardlessBproveBpregnantBoffendedBnumbersBlandBfemaleBdroppedBdeletedBchinaBbcBanimalBwinterBparkBnetflixBlatelyBinsultBhumorBdisappointingBcallsBawBstandardBshowedBrelevantB	instagramBglassBgeniusBfurtherBepicBbsB16BworkersBwantingBstartsBshotsBpettyB	nightmareBlonelyBjudgeBironyBforcedB
basketballBregularBperspectiveBpersonalityBpeoplesBmajorBjerseyBchicagoBbuttBbeyondBuselessBtrackBselfishB	potentialBpaperBmodsBlargeBdressBdirtyBchangesBbizarreBbabiesBtreatB	therapistBsuggestBrnBpointingBmonsterBmilitaryBlyingBholeBgorgeousBexistsBahhBabilityBunfortunateBtruckBskillsBprocessBopportunityBnoteBmagicBironicBgiantBfairlyBdudesBdirectlyBbuyingBbearBallowB3rdB14BwhetherBqbBhorrorBhelloBcarryBbraveB18BsmokeBshouldn’tBprivateBinjuredB
impressiveBhospitalBhatBexplanationBeBcopBcomplainingB11BtendBstreetBsongsBliberalsBlevelsBkillsBimmediatelyBhockeyBguessingBflyB	desperateBbloodyBbastardBanxiousB
understoodBseveralBroleB
restaurantBpolicyBpickedBmeatBmajorityBfeedBcreatedB35BwonderedBveganBupvotesBteachBsupposeBsignedBperformanceBnervousB
explainingBweakBrollBoohBnonsenseBjokingBhitsBgrewBfreshB	downvotesBdivorceBcameraBbandBawwBtoolBrussianBproBoverallBmaleBlessonBepisodesBcruelBcloserBbuttonBbecomingB	sarcasticBpropertyBkickBkeyBhideBfreedomBdestroyBconceptBashamedBsatireBpracticeBplanetBluckilyB	impressedBdislikeBchatBbritishBbaseBbBarguingBactingBtableBstudentsBrefsBpushingB
populationBpicsBphotosBpassedBfoolBfallingBbuildingBbringsBusefulBtheydBreleaseBrecentBrateBprovideBpetBownerBmodelBgoodnessBfourBfootBfacesBclaimsBappreciatedBtreeBswitchBsplitBspiritBshockBproperlyBproperBprayingB
physicallyBnationalB
horrifyingBherselfBheldBheadsBguiltyBgenderBfullyBfieldBerrorB	educationBdramaB	curiosityBcakedayBbeautyBarmBactiveBprogressBpopBmemoriesB	ignoranceB	depressedB	democracyBcoveredBclueBclothesBboxBbotherBwhoeverBvotesBtypicalBtexasBrelationshipsBplanningBmodernBlmfaoBlazyBit’llBgrownBfuckinBfrenchB	followingBfloridaBeuB2016BtonBsuitB	sufferingB	stupidityBrussiaB	referringBmodBmattersBhireB	destroyedB
depressingBdealingBbecomesBuponBtBshellBresultsBrepostB	offensiveBlawyerBjoyBgoalsBgfBflyingBdollarsBcoachingBviewsBteethBspeedBscrewBscoreBphonesBpantsBoptionsBlowerBlegBkingBimpactBhedBhasn’tBfamiliarBconfirmB
commentingBbasicB2018BwarsBvalidBuniqueBtalentBstudentBresponsibleBphraseBjusticeBhungryBgoldenB
disturbingB	challengeBbillBwindowBtopicBtoiletBthrownBspeechB
situationsBsidesBshowerBposterBpicksBmumBminimumBloserBflagBfBdeservedBbelowB😊BstayingBspecificallyBsarcasmBsaltBpushBnutsBhmmBdinnerBcrowdBcomedyBbanBalotB4thBwoahBwBtrollsBsaintsBresultBrB	ourselvesBorangeBmassBgloriousBextremeBeverydayB
delusionalBdegreeBdaddyBcreateBbankBscamB	offseasonBnamedBmealBkillerB	instantlyB
generationBfilledBemailBattacksBassaultBasideBaccidentBwokeBtweetBtalkedBsexistBmessedBmeetingBjourneyBiqBhittingBfingersB
downvotingBdistanceBdefBdataBchancesBbusBbecameBbathroomBantiBangerB	advantageB
would’veBtechnicallyB
supportersBstomachBsomewhatBsecurityBrevengeBprimeBpreviousBpcB
particularBmachineBliberalBhurtingBfilmBeventBdriverB	decisionsBbossBahhhB23B🤔BviolentBteenBsunBsmokingBslowlyBremindB	necessaryB	miserableBmapBlegendBheckBfishB	customersBcomfortableBcashBbruhBboredB	accordingBwho’sB
surprisingB
suggestionBspamBshipBpodcastBpartiesBincelBhardlyBelectionBdrinkingBdebateBdanceBabusiveB
unexpectedBtenBtaxesBstepsBstealingBstandingBremoveBranBquietBpoorlyBplotBnormallyBminBinputBfindsBexplainsBemotionsBelectedBdescribeBcrossB
commercialB21B2020BworryingBtinyBtheoryBspendingBshopBseasonsBretardedBputsBpreparedBlunchB
individualBhonestyBhandsomeBgymBfederalB	favoritesB	everytimeB	corporateBchargeBchangingBarticlesBwingBwineBuserBtreatedBsomeonesB	screamingBreplaceB	religiousBmirrorBimproveBgrowingBgifBequallyBembarrassedBdrawBcouldveBalcoholBviolenceBtallBsooooBsizeB	sensitiveBseatBsaltyBrespondBreachBpressureBpartsBoldsBjustifyBgotchaBfascinatingBfantasyBdriversBdrBcupBcreamBchoicesBcBbrownBbenchB🤣BweddingBunitedBtfBtankBsuckedBsleepingBpromiseBprogramBplsBmixedBmaterialBlimitBlaneBladiesB	includingBgradeBexcitingB
definitionBchoseBbottleBbenefitBattackedB90Bwe’veBtilB	standardsBsoldBsecretBpreciousBpotBpleasureBpassingBouttaBmanagedBlaidB	insultingBheroesBforeignBfitsB	financialBentitledBdisorderBcriedBconservativeBbattleBasapB
aggressiveB500BwhiningBvotingBsurgeryB
successfulBsauceBrangeBpressBontoBheavenBhadntBflatBfailureBenvironmentBdragBdowntownBbrexitBbombBblindBsportBsoloBskyBscumBriseBringBplateBpanicBnecessarilyBmatchesBlocationBitdBhorriblyBgoddamnBgainBearsBcitiesBbugBbagBbadassBappleB17By’allB
universityBstoleBstarsBstaffBsellingB
respondingBremainBrelaxBrefuseBplasticBmovesBmentallyBincomeBhardestBgermanyBfriendlyB	existenceBdunnoB	defensiveBchineseBcheaperBbreaksB	australiaByellowBsoftBrealiseB
propagandaBprofileBperhapsBpatchBmilkBgoatBgayBdangBbeginBwishesBvibesBusualBuniverseBturningBtellsBsoonerBplansBnatureBlBintenseBhumanityBhappierBgodsBfortniteBfasterBeffectsBdangerBcompetitionBcomicBcenterBcampaignBannoyedBthreadsB
supportingBstudyBspellingBsolutionBremotelyB	recognizeBpricesBplayoffBorganizationBnotesBmixBmiracleBlogoBkickedBhypeBhorseBheavyBdrawingBbearsBateB34BwiseBweatherBupvotedBstretchBstandsBsilverBshouldveBsavingBpullingBpowerfulBmisreadBlistenedBirelandBhangingBgroupsBfavorBexpectationsBenemyB	employeesBdoctorsB	confirmedBclimateBbreathBboatBvirginB
underratedBuncleBtrainingBsixBseekB	relatableB	realisticBpushedBpossibilityBoopsBnowhereBnailBlicenseBheatBguardBcrushBcriminalBcommitBclipBchargedBbringingBblessedBactionsB
thankfullyBtargetBstressB	pointlessBnoseBmovementBlabourBkissBinsightBhuntingB
healthcareBgoodbyeBfruitBempathyBdetroitBcringeyBcivilBcanadianBapplyB45BvictimBtwistBtradedBtongueBstruggleBremindedBolBoddlyBnativeBmistakesBliarBkindsBhatingB
harassmentBgrandmaBfilmingBedgyBdreamsBdecadeBdaBborderBboiBbillionBasleepBartistBamongBamazonBain’tB1000B	worthlessBvidBtrickBstrengthBsigningB
screenshotBscenarioBroadsBraisedB
pretendingBmarkBlionsBinjuriesBignoringBhatredBformerBffsBfallsBdocumentaryBdetailsB	convincedBbirdsBaffectB60Bwe’llBwastedBusesBtipsBsuspectBsurviveBscreamBrunsBrainBpetsBmurderedBknifeB
irrelevantBirlBinsecureBhopesBguiltBdecadesBdallasBconservativesB
comparisonB	communismBcastBbotheredBbelongsBbelievesB5thB😂😂BwishingBwhoseBwhoopsBwashBwalkedBvaluableBunbelievableBtalksBstreamB	sacrificeBreminderBprojectBnetBmehBmarchBlimitedBliedBholidayBhahahahaBgenuineBfreakB	franchiseBfingerBfascistBclarificationBcaringBburnedB	awarenessB	argumentsBaddressBabortionByou’dBwaitedB	unpopularBultimateBtriggerB	strangersBsourcesBshitpostBsaturdayBsafetyBruiningBrarelyBrageBopeningB	obnoxiousBmarryBjanuaryBincelsBimplyingBfoxBexperiencesBeditedBdollarBcultBcrackB	confusionB	confidentBcomebackBchampionshipB	celebrateBbubbleBblatantBautismBassumedBabusedB19B🙄B😭BwinnerBwhoaBvaginaButterlyBturkeyBtalentedBsufferBstopsBspellBseparateBrepliesBreleasedBrapedBprideBpresentBpoolBnflBmansBjeezBincludeBhenceBdonateBdirectBcreativeBcomboBcircleBbridgeBbirthB	believingBbeachBawhileBarrestedBaltBagentB24BwealthBwageBupdatedBthrewBstrongerBstatsBsmellsBsexuallyBserverBscenesBsadnessBrollingBraiseBpickingB	overratedBobsessedBnsfwBneighborhoodBmBisraelBforgiveBfactorBemptyB	elsewhereBdiscussBcutestBcuntBcostsBappearB70B1010ByBworldsBvibeBviaBvBtonsBteaB	socialismBshockingBschoolsB
satisfyingBrookieBrecoveryBprovenBpackBnowadaysBnightsBneckBloopBlightsBleavesBincreaseBhonorBgrabBgoshBflexBdoucheBdesignBdcBcustomerB	confusingBcheatBblamingBbeatingBaddingBactsB😅BthatllB	terrifiedBtagBstunningBsoooBshutdownBshiftBservicesBseesBscrewedBscareBrushBrejectedBreadsBproductsBoweBopenedBmomsBmexicoBlikingBkittyBkB	incorrectBhookBhoneyBhandledBfillBfeminismBexperiencedBembarrassmentBedgeB	correctlyBconsequencesBchuckleBchiefBbulletBbfBalternativeByorkBtoyBthrowsBtaughtBsugarBsubjectBsortsBsomedayBsignificantBrosterBromanticBrankedBps4BnailedBmemberBinspiredBindustryBillnessBheadlineBggBeuropeBengageBdmB
discoveredBdesignedBcringyBcombatBclickBchestBcheckingBcaredB
capitalismBbyeBbrandBbaseballBappropriateBanswersBaccountsB99B80B👍BweirdoBtinderBstoppingBstationBsmashBserveBseaBreportedBreplyingB	regardingB	realizingBreactB
punishmentBpairB
nightmaresBneatBlogicalBloadBlegallyBleaderBinstantBinappropriateBhe’dBhealingBfundingBformatBcrisisBcopyB
conspiracyBblowsBbeastBbastardsBauntBagesBabsurdB200B😉ByellingBwideBwastingBtimingBthxBthickBteachingB
subredditsBskillBrowBquarterBpplBofficialBnyBminorBmidBmaskBlinksBindiaBidentityBhiddenBhealBgasBgBfyiBentertainmentBeggB	describedBdamnitB	criticismBassholesBaimB	addictionB👏BwetBvictimsBupsBumBtypesBsmallerBshedBretailBrentBpreB
people’sB	neighborsBmoodBislandBinsultsBhillBgmBframeBfoulBfarmB	everyonesBemotionallyBegoBearBdisappointmentBcursedBcookingBcoastB
californiaBbudgetBbayBanytimeBanimeBanalogyBactivelyBunironicallyB	typicallyBtoneBthemeBstayedBsolveBsociallyBseatsBrippedB	remindingBrapBprogressiveBpityBouchBooohB	obsessionBnycBmillionsBlooseBlegitimatelyB
legitimateBkitchenBimaginedBiircBhmmmBhappilyBgamingBexchangeBdrivesBdrinksBdiseaseBdisasterB	democratsB
democraticBcurseB	correctedB	childhoodBbitterBbeardBatmBamazedB2017BzoneBwarningBvolumeBunionBthreatB	thereforeBtattooB
supportiveBsundayBsueBstrikeBstBsouthernBsignsBshoulderBshapeBreliefBpovertyBparticularlyBnetworkBmindsBmetalBmathBleadingBlameBjapanB	influenceBimmoralBhellaBgrowthBflipBfamBdozensBcuttingB	creaturesBcombinedB	chocolateBchiefsBcausingBbootsBblowBblockedBbangBbalanceBbaitBarizonaBactorB90sBworkerBweaponBvaccinesBvacationButterBunlikelyBunderestimateBticketB	thousandsBterriblyBsquadBslaveryBsightBsharedBsectionBscoredBscaresBridingBretireB	responsesBpeeBpalBownersB
optimisticBmountainBmedsBmanagerBkindnessBjamBintelligentBheavilyBharshBhardcoreBfoBeveningBenglandBempireBeliteB	effectiveBdisneyBcrimesBcrewBcounterBcornerBconversationsB	companiesBclappedBbibleBbiasBbellB	amazinglyBaccessByehBthatdBspoiledBsmugBsettingBscoringBridiculouslyBrationalBpunchBpoliticiansBpinkBorderedBofficerBnhlBnationBmondayBmessingBmessagesBmembersBmediocreBmedicineBmamaBlinkedBletterBleafsBknockBjumpingBintentionallyBignoredBhahBfunnierBforcingBfashionBfamousBenvyBdryBcoreBconsistentlyBcenturyBbothersB	backwardsB
assumptionBadsB31B😂😂😂B	weren’tBuberBtreatsBthusBstoneBstatedBsolvedBslapBskipB
retirementBrecommendationBrankBpraiseBpatienceBpastaBparanoidBopsBocBmoonB
misleadingBmasterBlibertarianBjetsBinvestigationB
homophobicBgrammarBgameplayBftfyB
friendshipBeyebrowsBeconomyBdetailedBdesperatelyBdammitB
conclusionB	comparingBcolorsB	chemistryBcasesBburningBbreadBblastBappearsB	apartmentBaccentByellBxboxBsweatyBstolenBspreadB	socialistB	sentimentBsemiB	reportingBrealisedBrantBracistsBpunBpulledB	positionsBplantBpillBpatientBnuclearBnorBnicerBmoralBmicBmagaBlegionBlakeBkneesBincludedBgutBgenBfollowedB
flashbacksBfascismB	executionBcookiesBconsentBcompetitiveBchargesBcartoonBbrightBblownBaudienceB	attractedBarmyB
appearanceB26B“theB	treatmentBtrailerBstockBstanceBsometimeBservedBroundsB
revolutionBrequireBrepeatBreceivedB	reasoningBprovidedBpiecesBpatsBpackersBnaziB	naturallyBmistakenBminsBmindsetBlossesBkillersBimmatureBhomelessBfamiliesBfailingBfaBeatenBeastBdoorsBdoggoBdietBconsoleBcodeBbullyBbullsBblanketBbeliefsBartistsBapproachBaffectedB😍BworthyBtweetsB	they’veBtdBsquareBshownB
preferenceBpopcornBplzBontarioBnovemberBnoiseB
medicationBmanageBlosesBlockBlilBinternationalBindependentBideologyBhighestBgoogledBghostBgenocideBgangBfilthyBfeedbackBfedBequalB	enjoyableBdancingBdamnedBcrappyBcorruptB	conditionBclosedBcircumstancesBchefBburgersBbirdBautoBaspectB7thByeBwindowsButahBturdBtrumpBtrendBtranslationBticketsB	they’llBteachersBstreakBstageBsonsBshiteBsecretlyBsecBruralBrootingBpriorBplaneBoursBopenlyBoblivionBnurseBmisunderstandingBminorityBmetaBleaningBjudgingBhorrificBholesBhere’sBhahahBguessedBgeezB	finishingBexitBessentiallyB	elaborateBdroppingBdodgedBdiscordBdemsB	defendingBcreepB
connectionBchairBcapBcaloriesBbumpBbullBboutBbikeBbellyBairportB
afterwardsB—BwesternBwalmartB	underwearBunderstandableBummBtypingBtriesBtradingBthousandBthankfulB
suggestingBsubtleBstormBspitBsinBprayersBpercentBpaysBpathBoccasionallyBnerdBmommyBmatureBmaintainBlondonBlockedBlipBleadsBhornyBheartbreakingBglassesB
frustratedBeraBearnBduckBdisrespectfulB	diagnosedBdescriptionBdebtB
correctionBconvinceB
complimentBcomfortBblewBautisticBattemptsB	answeringBwordedB
vaccinatedBunnecessaryBunhappyBtravelBtortureBteenagerBtechBtearBsyndromeBswordBswitchedBsignificantlyBshitsBservingBscreamsBreverseB
relativelyBreachingBratedBramsBpurpleBpumpBpaintingBownedB
officiallyB	motivatedBkindlyB	imaginingBhatefulBhandedBhamBfurryBflopBfetishBfbiBexpressB	explainedB
excitementB	emergencyBdiscriminationB
departmentBdeeperBconstantBcondescendingBcoltsBcleverBchoosingB	celebrityBcausesB	candidateB
boundariesBbonusBboneBbodiesBaweB	attackingBasiaBareasBaltrightBahaB
acceptableB22BwindBwifesBweirdlyBvagueBupperBtorontoBthinBtheftB
terroristsBtastyBtaggedBswingBsurvivedB
statementsBspanishBsingB	sincerelyBshaveBshallBseattleBscotlandBscheduleBscaleBsandBrootBrecallBprovesBprimaryBotB	narrativeBmoronsBmilesBmeasureB	marketingBmacBjuiceBitemBinchesB	hypocrisyBhomesBfunctionB	forgottenBfeedingBfallenBexamplesBdragonB	directionBdeskBcomradeBcomputerBcombinationBcentralBcatchingBcaptainBbuffB	breathingBautomaticallyB	anarchistBadditionBacceptedB	abandonedB95B27ByahBwomensB
whatsoeverBunattractiveBtoothBthursdayBsuddenBstrategyB	spreadingBspinBsighBserialBreviewB	regularlyBregretsB
previouslyBprBpoopBpeakB	overwatchBopportunitiesBohhhBnaiveBmisunderstoodBmanipulativeB
managementBmahB	legendaryBlaughsBkoreaBkneeBjunkBjoinedB
ironicallyBinteractionBintendedB
immigrantsBgreedyBgenericBfriesB	feministsBfeministBfancyBencouragingBdumpBdumbestB
disrespectBdelusionBdefenderBcondolencesBcompareBcommunitiesBcluelessBchildishBbigotBbangingBatheistBagedBaccusedBwheresBvisibleBunluckyBtragicBtaskBsymptomsB	supporterBsuggestionsB
subscribedBstudiesB
strugglingBstreetsBsenateBrifleBradioB
protectionBposBpassiveBoverwhelmingBofferedBnetsBnbaBlayBintjBinformativeBhersB	guaranteeBgreaterBflightBfavBexposedBeffectivelyBeconomicB	disgustedBdigB	developedBdecemberBdeBcubsBcommuteB	communistB
collectionB
clarifyingBcitizensBcampBburgerBbenefitsBbeltB	behaviourBbbBwisdomBwireBweaponsBvegasBunderageBunableB
tournamentBtoddlerBtimelineBtankingBsympathyBstickingBstarterBsizedBshooterBshadyBseekingBrequiredB	relativesBprofitBoutcomeBohioBmessyBmeaninglessBmaxBittBintroBinfuriatingBimmigrationBicingBhypedBhotelB
horrendousB	highlightBhandlingBhallBfridayB
frequentlyB
equivalentBeggsBeaglesBdramaticBdraftedBdopeBdatedBcropB	continuesBcomplicatedBclassesBcitizenBchaseBcardsBcapableBbrothersBbeatsBadministrationBvikingsB	unrelatedBunlikeBunitBtrapB	translateBtragedyBtljB
technologyBsurprisinglyBstatusBstarvingBspotsBspeakerBsoundedBsoulsBskinnyBsensibleB
republicanBrepostedB
rememberedB	receivingBproductB	privilegeBpocketsBphaseBpanBohhBoffersBneighborBncBnapB	minnesotaBmindedBiphoneBintelligenceBimprovementBikBidentifyBharmBgriefBgladlyBgigBgearBfrustrationBflagsBfbB	exceptionBescapeBencounteredBelsesBeducatedBdeliveryBcriticalBcreatingBcousinBcouplesBcountyB
comprehendB	committedBcoincidenceBclothingBchelseaBcheerBchecksB
capitalistBcancelBboomBboobsBbillsBbaconB	assaultedBactorsB😢B😁BytByou”BwelpBwaveBvileBventB	venezuelaBunusualBtowardBtightB	territoryB	stressfulB	streamingBspeaksBsnapBsnakeBsloppyBskinsBsinkBsharkBschemeBsaleB	righteousBrepBremindmeBrelievedB
recognizedBrabbitBpuckBpubliclyBpodsBplugBpillsBopensBname”BmutualBmortyBmildB	mentalityB
meaningfulBlotteryB	lifestyleBlickB	incapableBheightBhatersBhaircutBfrozenBfacingB
expressionBeventsBendlessBemojiBeatsBdifferentlyBcycleBcrashBcommunicationBcommunicateBclarifyBbummerBbpdBbitsBbingedBbelievedBbagsBbackupBapBantisemitismBankleBagencyB	acceptingBwreckBwoodBwalletBvictoryBusersBtwinsBtwentyBtubeBtributeBtheoriesB	teenagersBstadiumBstabbedBspokenBsleptBsitesBsistersBshould’veBsheepBshamingBscratchBrouteBrequestBrefusingB
refreshingBquestionableBpuppyBprovingBpropsB
projectingBpleasedBpigBpenaltyBordersBohhhhhBnonethelessBmoralityBmoderateBlaunchBjerseysBinsanelyBhoustonBhopefulB
highlightsBhidingBheartsBhammerBgraveBgrandBfirmBfilmsBfightsBexcusesB	exclusiveB	encourageBearnedB
discussingBdevelopmentBdevelopB
destroyingBdesireBdenyBdeerBcrownBcountsB
correctingB
comfortingBcoinBclickedBchickBcheatedBcaptionBbuiltBbrownsBbipolarBbeingsBbeansBbasementBbarrelBaxeBasksBappliedBafricaBadvanceBadoreB38B33B❤BweirderBwallsBvisitingBtyBtrophyB	teammatesBtapeBsupplyB	superbowlB
subjectiveB	strugglesBsocksBrumorsBriceBrewardBrememberingBremainsB
referendumB	redditorsBrecordsBreachedBpoundsBpoliciesBplatformBplainBpitchBpersonsBpassesBobjectivelyBnattyBmysteryBmuscleBmissionBmintBmetsB	marijuanaBlockerBlegoBincludesBhousesBhomoBhappiestBguideBgreatlyBforumBflowersBflashBfinanciallyBfileBfartBexperiencingBembraceBdropsBdemocratBcrossingBcowardBcontroversialB
controllerB
controlledBconstitutionBconsiderationBcokeBbucksBbreedBbreatheBbonesBbonerBbioBbatBauthoritarianBaudioBarrowBarrogantBarmorBanalysisB	admittingB56B😔B“B
washingtonBvehicleBvaluesBvalleyBtreasureBtrashyBtransferBtierBthreatsBthreateningBthreatenBtestedB
suspiciousBspeciesBsandwichBsaferBsaddestBrolesBrobotBrewatchBresponsibilityB	repulsiveBremovingBreBratesB	providingBprogramsBproducedBppB
politicianBpleasantBpenBpassionBoverlyBoutfitBorlandoBoddsBlineupBlargelyBlakersBkitBit’dB
inevitableBimprovedBhowsBholidaysBhatsBhadn’tB
guaranteedBgreasyBgratefulBgrassBgovernmentsBgiantsBfunniestBfoodsBflowBexerciseBeducateBdistantB	discussedBdevilBdevastatingBdestructionBdelightBcoworkerBcowboysB
conditionsBcmonBcloneBclearedBchipBcharmingBchargersBcarriedBcaBburdenBbrosBbroncosBbotsBbostonBbchBbackedBatleastBamountsBalabamaBaccomplishmentB300B“youB¯ツ¯BworeBvoidBtypedBtouchingBtouchedB
throughoutBsydneyBsurvivalBstuntBspicyBsoooooBsoberBslamBsilentBshineBscumbagBscoresBrollsBrocketB	respondedB	resourcesBrepublicansB	representBregisterBpupBpsychoBprequelBpolyBpillowsBpatriotsBpartnersBpanelBoccurredBnakedBmockBmethodBmercyBmavsBmarvelB
mainstreamBmafiaBloyalBlitBlayingBlastedBkudosBkickingBjapaneseBitemsB	interestsBinstinctB
impressionBhuntBhunBhomieBhmBgullibleBgraspBfeesBfebruaryBenjoysBdumpedBdocBdiggingBdiesBdevsBdeviceBdetailBdemandBdefeatBcuckB
conferenceB
complaintsBcompBclownBclosureB	cancelledBbutterBbiteBbitcoinBbedroomBbasketBbarsB
australianBattorneyBangleBamenBallowingBaffairB
advocatingB
additionalB43B29B“ohBwoundBwittyBwingersBwarnedBvisitedB	virginityB	unhealthyBunfairBtireBthruBterrorBsyriaBsweetieBsteelBstaysBsodaBsingerBshoveB	shelteredBshadeB	sexualityBsetsB	searchingBseafoodBsbBsackBrespectsBrepresentationB
referencesBredditorBrealizationBrawBrapingBraisingBpupperBpsBpopsBpharmaBpatientsBpatBpaintB	obliviousBnikeBnervesBmvpBmouseBmorallyBmenuBlistedBladBknicksBjuicyB
journalistB	intriguedBinitialBhundredsBhotsBhiredBhehBhappyfriendlybotBfragileBfoulsBflamesBfeatureBfascistsBexistingBespnBeasternB	dependingBdeclineBdeckBdashB	criticizeBcoughBcoolestB
concerningB	complaintBcolourBclearingBclaimingBbumperBbrainwashedBbillionsBbbcBbasisBarsenalBaprilBaocBansweredB
activitiesB80sB“nameByardBwitchBweirdestB	wednesdayBwagesB	vancouverB	upsettingBupdatesBturtlesB	that’llBtestingBsystemsB	switchingBsupportsBstronglyBstatingBspringBsmoothBsimultaneouslyBsfBsavesBsavageBriverBriddanceBrestaurantsB	religionsBrecipeBrapistsBradicalBprovedBprotestBporkBpoliteBpolishBpointedBpipeBpileBpermanentlyBparanoiaBopposedB	opponentsBnpBmonkeysBmonkeyBmockingBmichiganBlatterBlaptopBinspirationBhideousBhandyBgentleBgenerationsBgemBfriedBflirtingBfestivalBeveryone’sBenteredBenforceBdoordashBdonutBdenialBdemBdefendedBdeeplyBdealsBdarnBcourageBcouncilBcouchBconvertB
committingBcollectBclubsBclosingBceilingBcartBbulliedBbuddiesB	breakfastBbootBblockingBbelongBbeliefB
beforehandBawardBauthorBasylumBarrivedBarrestBantivaxxersB	addressedBacidB20sB😎B♥️B	wikipediaBwedBtsaBtripsBtrippingBtrilogyBtheirsBtacosBtacoBsurvivorBsurveyB
surroundedB
supposedlyBsunshineB	subscribeBstrawmanBsteakB
shockinglyBsequelB	sentencesBscottishBsanBsalesBrumorBrepliedBpunchedBprospectBprintBpowersBpocketBpingB	permanentB	pedophileB	paragraphBpagesBpadBoffendBnutBnooooBnerfBmustveBmmmmBmasculinityB	maliciousBlowkeyBlongestBlgbtBleanB
leadershipBlatestBlaborBkinkyBisisBintpB
intolerantB	immenselyBigBidealBhumourBhumbleBheelsBgreyBgilletteBgamersBfourthBfishingBfacialB	efficientB	eachotherBdownhillBdonutsB	demandingB	daughtersBdadsBcustomBculturalBcryptoBcopeBconnectBcomplexB
commitmentBcloudsBcleaningBchampBchainBbugsBbackingB
astoundingBasianBarguablyBapplaudBalertBahhhhBadoptBacknowledgeB4chanB36B2000ByankeesBwoodsB	wisconsinB	unethicalBundergroundBunBtypoBsweetestB	supportedBsuccessfullyBspareBspammingBsortedBskullBsingingBsevereBserversB	selectionB	scrollingBsalaryBsaladBroastingBrfunnyBretiredB
resolutionBrejectBratingBquotingBpreyBpredictBpracticallyB
percentageBpaBovenBorgBnoobBmiseryBmildlyBmethBmanipulationBleftistsBleftistBkingsB	justifiedBjumpedBitalianBiowaB	invisibleB
introducedBimagesBhousingBhostBhoBhipsterBheadedBhalfwayBhahahahBgrandpaBgoalieBgermanB	furnitureBfiguresBexamB
enthusiasmBencouragementBemployeeBedBdumberBdreadBdivisionBdesktopB
describingB	deliveredBcynicalBcreatesBcourtsB
could’veBcontestBconsistencyBcongressBcondomsBcomfyBcnnBclapBchokedBchipsBchillsBbullyingBblowingBbitesB
biologicalB
believableBbattlesBaustinB
attractionB	attemptedBaswellBappealBamusingBaddsB30sB2014B🙃B😆BworshipBwingsBwelfareBwakingBvotersBvanillaBtransitBthey’dBteBswallowBstrokeB	strangelyBstanBspoilerBsoreBsolelyB
socialistsBslipperyB	slightestBsiblingsBshopsBshiningBscriptBsaneBromanceBreliableBrecoverBrazorB
psychopathB	promotionB
productionBprickBprepareBpistonsBpissingBpeepsBoutrageBopponentBnailsBmuseumBmotherfuckingBmotherfuckerBmetaphorBmediumB	mcdonaldsBmailBliteralBlimitsBlifetimeBlapBjkBjanBinviteBinnerBinchB
inaccurateB	inabilityB
importanceB	imaginaryBhuluBhrBhawksBhabitBgamerBfrigginBfreakyBfocusedBfloatingBexpandBemotionBeffortsBdynamicBdodgeBdiveBdialogueBdevilsBdenyingB
delightfulBdecidesBdebatingBdamB	crossoverB	coworkersBcookB
convenientBcoloradoBcollapseBchinBcbB
candidatesBcampusBburstBbrushBbrowsingBbrooklynBbravoBbrainwashingBbpBboozeBbigotryBasexualBarseBarcBapproveB	announcedBancientBammoBallergicBaddictedB41B2009B	‍♀️ByuckByrsBwebBwashedB
vulnerableBviceBvehiclesBunreasonableBtwinBtunedB	traditionBthroatB
thoughtfulBtextsBteensBtanksBsuedBsubwayBstressedB	statisticBspokeBspoilBspeculationBsmellingBsmarterBshillBshamefulBshadowBsevenBscarsBsatisfactionBsampleBroofBrequiresBraptorsBrainbowB	qualifiedBpunkB
protestingB	pregnancyBpotatoB
possessionBphrasedB	parentingBorgansBooooBobservationBobscureB	objectiveB	must’veBmomentumBmlsBmisogynyBmilBmidnightB	melbourneBmeaslesB	magicallyBm8BlegendsBlargerBkissingBkeysBkatBintimidatingB	intentionB
insightfulBincidentB	householdBhormonesBhipBhe’llBharassBhackBgrandparentsBgrandmotherBgopBgloryBghostedBfundBfoughtBfluBfictionBfiancéBfarmersBexpertB
exhaustingBerBdumpsterBdroveBdishBdisgraceBdisabledBdenverBdentistB	dedicatedBddBcrossedBcountingBcontributingBconfirmationBcloselyBclaimedBchosenBcharityBccBcapitalistsB
borderlineBbetrayedBbegBbathBbashingBbakedBartifactBappointmentBapplyingBandroidBanalBalternativesBadoptedBaceB250B150BzombieByknowBwipeB	where’sBwe’dBwarmingBvomitBvoicesBvarietyBvalveBunicornBtumblrBtrucksB	travelingBtossB
thoroughlyBtastesBtastedBtampaBswimmingB	surprisesBsuitsB	structureBsteelersBspineBsooBsneakyBshoutingBshortsBshoppingB	shitpostsBsettleBscrollBscrewingBsaudiBsanityBribsBrewindBreviewsBresistB
repeatedlyBreferBrefBreckonBrecklessBreceiveBrecB	reactionsBratBquestioningBqcxB	preferredBpitBpieB
passionateBparodyBoutrightB
oppositionBooBomfgBnorthernBnonexistentBnjBnicelyB	nevermindBneedingBnarcBmothersBmlmBmisinformationBminiBmedB	meanwhileBmatesBmagicalBlosersBlanesBketchupBjuulBjudgesBjeansBjacketB
irritatingBintentionalB	hypocriteB
highschoolB	headlinesBheadingBgraphicBgrantBgoatsBglovesBgapBfreakinBfranklyBfrankBforthBfootageBfinalsBfellaB	exhaustedBeternityBestablishedBequalityBengagedBelectricBdustBdoomedBdonationBdnaBdlcBdifferencesBdiarrheaBdestroysBdepthBdeliberatelyBcureBcreepsBcredibilityBcousinsBcomicsB	cognitiveBcodBcarrotBcapitalBbrowsBboxingBbleedB	blatantlyBbladeBbiasedBbeefBbaselessBbananaBasshatBarenaBapologyBannoyBanniversaryB	amendmentBalasBaimingBagendaB	adventureB400B	“it’sBwritersB
withdrawalBwheelBvrBvisaBvariousBvanityBunacceptableB
toothpasteBtoolsBtitlesBtiresBthumbB	terrorismBtemperatureBtadBsunsetB	stupidestB
stereotypeBsteamB	spiritualB	snowflakeBslideBsilenceBsignalsBshoutBshotgunBshookBshippingBshe’dBshavingBsecureBsafelyBroseBroommateBrevealB
reflectionBreaperBratsBrandomlyBramblingBrainyB
projectionBprobBprizeBprankBpotentiallyBpokemonBpoetryBpodcastsBpissyBpantiesBovercomeBoregonBofferingBo7BnursesBnudeBnglBnfcB	neckbeardBmpB	mountainsB
mentioningB
masturbateBlipsB	lightningBlibBlettersBledBlackingBknockedBjawBinvitedB	initiallyBinformBinconsistentBieBhypocriticalBhurryBhsBhobbyB
historicalBheatedBheadacheBguitarBgrindBgirlfriendsBgbBfundedBfrighteningBfreezingBfreaksB	followersBfinaleBfighterBfateBexpertsBeternalBdlB
determinedBdesperationBdegreesBdeclareBdealtBcringedBcreatorBcraftBcoupBcostumeBcorrelationBcornB
contributeB	contractsBconcertB
compellingB	commentedBcomedicBcoalB
cigarettesBchamberBcarryingBbulliesBbtcBbooBbombsBbiologyBbingoBbeautifullyBbeatenBbasesBavoidingB	authorityB	appealingBanarchyB	alternateBalarmBaffectsBadmittedB	accidentsB4kB20kB2012B😐ByrBwutBwolfBweeklyBwarriorsBviewersBveggiesBvastBvaccineB	vaccinateBusageBunsureBtuesdayBtripleBtreatingBtransBtoxicityBtheaterBthat’dBtextingBterminalBtattoosBsummaryB	strongestB
statisticsBstatBstartersB	spidermanBsmashedBskyrimBsketchyBshortlyBshootoutBshakingBshakeBsabresBreplacementBrelyBrefusedBreceiverBreactingBqueensBquartersBpunchingB	publicityB	protectedBpromoteBprofitsB	producersBpreventBpretentiousBplacedBpikachuB	piercingsBphillyBpeersBpapersBpadresBowningB
oppressionBnaBmythBmommaB	minecraftB
millennialBmidwestBmiamiBlookinBlabelBknocksBkittensBjealousyBjabBitalyBinfrastructureBindividualsBindianB	increasedBharassedBgtfoB	graduatedBgooglingBgatekeepingBfurB
forgettingBfoolsBfixingBfactorsBfacedBexsBexclusivelyBestablishmentBedmontonB	economicsBducksB	douchebagBdoubtfulBdisingenuousBdirectorB
directionsBdipBdespiseBdefineBdefaultBcuredBcreepedBcowBconvenientlyB
continuingB
contactingBconclusionsBconBclemsonBclassyBcarbonBbuzzBbreakupBboundBblogBbeersBbanningBbanksBawakeB	atrociousB	animationBalienBagentsBadmireBachieveBaccusationsB
accusationBabusingB8thB47B2013B2010B2005B“whatBwriterBwrapBwolvesBwitnessBwhataboutismBvomitingB
visibilityBviciousB
validationBuploadedBunrealB
uninformedB
unemployedB
ultimatelyBufcBtrickedB
transitionBtoysBtourB	toleranceBtoeBtippedBtieBtideBsushiBsurroundingBsucceedBsubmitBstinkB	starbucksBspottedBspoilersBsoupBsorrowBsolvesBsoccerBsnortingBshorterBshootsBshootersBsharpB
scientistsBroastBriotB
respectfulBresourceBrescueBremakeBrefusesB
referencedBranchBquietlyBpressingBportionBpoppedBpleasingBplaythroughB	placementBpillowB	photoshopBperformancesBparisBowoB	ownershipBoutragedBordinaryBopposingBogBodysseyBnotedB
negativityBnawBnauseousBnarcissisticB
motivationBmorbidBmissesBmetricBmapsBloanBlikeableBlibraryBlibertariansBleatherBkoreanBknightsBjulyBjudgedBjournalistsBintellectualB	inspiringB
inherentlyB	incentiveB	illegallyBhusbandsBhopBhogBhesheBhealsBharmfulB	harassingBguestBgrantedBgenerousBfurriesBflowerBfledBflawsBfilthB	expansionB	evolutionB	essentialBdutchBdullBdraftingBdodgersBdestinyBdelayBcyclingBcurbBcuddlingBcravingBcrackedBcowardsBcontrollingB
continuousBconfirmsB
confirmingBcomprehensionBcommieBcommentatorB
commentaryBclockBclappingB	chemicalsBcentreBcellBcapacityBbuzzfeedBbragBblamedBbingeBbiBbetrayBbetaBbeggingBbeeBassistsBassetsBarmedBappliesB	anecdotalBamusedBamiriteBamazesBaimedB	agreementB	afternoonBadvertisingB59B510B39B1200B101B10000B“notB͡°ByerBwthBwsBwoundsBwondersBwizardBwinkBwhoreBwhereverBwearsBwalksBvisualBvisionBvietnamBvetB	upliftingBunseeBuniteBunemploymentBunderstandsBtwelveB
triggeringBtraditionalBtractionBtrackingBtowerBtopsBtldrBtensionBteenageBtbfBsuckersB	subtitlesBsubscribersBstokedBsteppedBstaringBssBsnackBsmilesBsinsBshyB	shouldersBshipsBshinyB	separatedBsectionsBscarBsammyBrulingBruinsBroyalBroutineBrobbedBrippingBrickBrevealedBresidentBreplacedBrelaxingB	relationsBreflectBrecreationalBrbBrapidBracialBpursueBpromoBprofessionalsB
presidentsBprepB	practicesBpostersBpissesBphysicsBphoenixBpharmacyBpetaBpersonaBpeakedBpaymentsBpaymentBparkingBorleansBofficersBniB	nashvilleBmlkB
membershipBmasterpieceBmarkedBmangaBmainlyBmachinesBlptBlowestBlootBlocatedBlistsBlinkingBliningBlighterBlengthBleadersBleBlazinessBkissedBkiddoBjazzB	involvingBinvestBintoleranceB
interviewsB
internallyBinsistBinsensitiveB
inequalityBindicateBimmortalBidentifyingBhometownB	hollywoodBhintBheheB	healthierBhandfulB
gymnasticsBgtaBgrowsBgripBgrayBgoblinBgalaxyBfuriousBfuelBfranceBforumsBflavorBfinestBfilterBfearsBfckingB	fantasizeBesteemBenormousBemailsBelBehhhBeditingBeaBduhBdressingBdoseBdominantBdollBdocumentB
dissonanceB	disappearBdesignsBdentalBdefensivelyB	defendersBdairyBcuntsBcuddleBcoverageB
conventionBcontainB
compromiseBcommercialsBclutchBchaosBcentsBcamBbustedBbreastBbowBbounceB	botheringBboldBbinBbeginsBbeganBattractBapplesBallowsB	addictiveB6thB69B2011B10kB🤢B😂😂😂😂B👌B👀B☺️ByummyBww2B
witnessingBwithdrawalsBwishfulBwheelsBwealthyBwatchesBwarnB
vegetablesBupsideB
unsettlingB
unpleasantBuncommonBtwitchBtrustingBtrustedBtiedBthotBthankyouBtennisBtendsBtacticsBsurrealB
superpowerBsuperiorBsunnyBstruckBstackedBspiderBspelledB
soundtrackBsomeone’sBsneezeBsneakBsliceBslangB	similarlyBsiblingB	showeringBshe’llBshamedBscotchB	scientistB	scenariosBscammingBrubBroomsBromeBrepresentativeBreplayBrelatesBregionBreducedB	recordingBrecognizingBranksBrackBqualifyBpvpB	punishingB
presidencyBpreachBprayerB
practicingBpokerBpokeBpodBplannedBpirateBpilotB	perceivedBpepperBpapaB
outrageousB	oppressedBoceanBnormBnonoBneedlesBnazisBmustacheBmusclesBmuhBmugBmtvBmsBmoralsBme”BmexicanB	metallicaB	memorableBloyaltyBloverBloreB	liabilityBlegislationBkongBkingdomBjackassBjackBinsanityBinformedBimpliedBikrB
hystericalBhypotheticalBhundredBhorridBhopedBhomeworkBhobbiesBhairlineBgorillaBgenreBgawdBgagBforeheadBforcesBfoolishBflamingBfirmlyBfiringBfeverBfeaturesBfathersBetBentityBenforcedB	electionsBelderlyBdxmBdutyBdreamingBdrainBdraggingBdoomBdividedBditchBdistressBdistractBdislikedBdisgustBdiscoBdirectedBdiceB	diagnosisBdevBdecidingBdatesBcriticizingBcriesBcosB
consistentB
concussionBconcernsBcommiesBcoachesBcoB
circlejerkBchronicBcheeredBcheekyBcheaterBchangerBcannonBcamerasB	buildingsBbravesBboopBblissBblahB	bathroomsBapproachingBapplicationB
announcersBandorB	anarchismBalphaBalaskaBagreesB
admittedlyBaccentsB600B3kB2015B2003BwordingB	willinglyB	willfullyBwhinyBwhilstBweighBwarrantB	volunteerB	violatingBvillainBversaBverbalBunrealisticBunheardBunconditionalBtwatBtroublesBtreesBtracksBtodaysBtoastBtempleBtaxedBtargetsBtalesBsweptBsurgeonBsuggestsB
sufficientBstripBstickyBsteppingBstealsBstareBsrsBspidersBspendsBsortaB	sociopathBsoapBsnapchatBsnacksBsmellyBslopeBskewedB	skepticalBsjwsBsitsBsimpsonsBshittingBsexismBseverelyBsensesBsecondedBseBsassyBrubbishBrolledBrobotsBreturnedBrethinkBresolvedB	repeatingB
remarkableBrehabBreduceBrecordedBratingsBquotesBquittingBqbsBpsychologicalBproteinBpromisedB
prioritiesBprincessB	presentedB	preciselyBpoliticallyBplatesBpencilB
pedophiliaB
parliamentBpaintedBoxfordBowBoptimismBohhhhBoccurBobeseBnipplesBnewbornBneglectBndpBmusicalBmisogynisticB	miserablyBminesBmillennialsBmerelyB	massivelyBmallB	ludicrousBloversBlogBlocksBliftBliarsBlegacyBlectureBlaughterB	laughableB	languagesBlabBjohnBislesBironBirishBinvestedBinventBinterpretationBinsecuritiesB
innovativeB	infectionBimpliesBimmaBhutBhow’dBhorsesBhonoredBhangsBhagarBhBgroceryBgrimBgradBgpBgovtBgoodsBgimmeBgeorgiaBfrogBfridgeBfrickBfreezeBfreelyBfreakedB
foundationBforkBforbidBfooledBfondBfluffyBfliesBfittingBfistBfewerBfestBfeedsBfameBfadeBexpenseBexesBexcessBeveBeloB
eliminatedBeasiestBdownloadBdivorcedBdistrictBdismissB	dishonestB	disagreesBdinosaurBdemonizeBdelayedBdeathsBdeafBdatBcrueltyBcrowdedBcriteriaBcrackingBcowboyB	countlessBconvoBconductB	completedB
communistsB	clarifiedBchuckledB	childrensBchicksBchallengingBcgiB	centristsBcandyBbustBburntBburnsBburialBbumBbucketBbrickBbreakerB
boyfriendsBbehaveBbehalfBbbqBbashBbarriersBbamBavatarB
associatedBassesBarriveBaquariaBaptBapparentBanyonesBannoysBangelBamendB	ambulanceBaiBactivityB	activistsB
accuratelyBaccomplishedB
accomplishBabBaaB911B75B44B3dB3000B😄BzionByouthByesssssBwornBweaknessBveteranBveinsBurgesBupvotingB
unfamiliarB
uneducatedBunclearBturkishBtuneBtubBtrivialBtraitorBtantrumBswedishB
suspicionsBsuprisedB	superstarBsunsBsummedBstudioBstrikesBsticksBstatisticalBstalkingBstableBspursBspouseB	specifiedBsparkBsovietBsoldiersBsmilingBslowerBsidewaysBsidewalkBsidebarBshitshowBshitpostingBshirtsBsheerB	shamelessBsetupBservesBsegmentB	seeminglyBsdBscrubB
scientificBsatBsantaBsaBrubberB	rpoliticsBridesBrhockeyBrespectableB
representsBreportsBrepeatedBregardB
recoveringBrealmB	rationaleBratioBrampantBradBquestBpurelyBpumpedBpuddleBptsdBpsychiatristB	promisingB
profitableBpricksBpresentsBpredictableB	practicalBpoundBposeBportalBpicklesBphrasingBpasteBpalmBpairingBoutstandingBoutlookBoughtBorthodoxBoriginsB	organizedBorderingBonionBoleBoilsBobjectBnovelBnotificationBnicotineBmiceBmeltsBmealsBmdB
manipulateBmaniaB
manchesterBmanagersBmaamB
lonelinessBloansBloadedBlizardBlessonsBleaksBleakedBlandsBknightB	kidnappedBkeeperB	judgementBjoiningBixBivBisraeliBisolatedBinvolveBintimacyBinferiorB
industrialB
indicationBholdsB	heartlessBhayBhairyBgutsBgreekBgoofyBgiggleBgeeBgainingBfrequentBforwardsB
foreignersBflavorsBfenceBfailsBfabulousB
experimentBexecutedBerrorsBengineeringBenforcementBemployedB	eliminateBechoBdrownBdreadfulBdozenBdonaldB	dominanceBdiscoverB
disappointBdemographicB	delightedB
dedicationBdarlingBdakotaBcrotchBcrashedBcorporationsB	copyrightB
conversionBcontributedBcommandBcoinsBchokeBchillingBchiliBcheeksBchasingBchartBcentristBcelticBceasesBcasualBcanceledBcanadasBbuttsBburiedBbudsB	brutalityBbroadBbritainBbraBboostBblondeBbigotedB
beneficialBbeesBbandsB
bamboozledBbalancedB	automaticBapprovedBapprovalBappreciatesBannounceBanecdoteBamdBalBaidsBahahaBagreeingBaggressivelyB
aggressionBacneBaccuseB48B42B2008B🙏B💕B“thankB’B͜ʖBwritesBwouldaBwivesB	witnessedBwildlyBwhaleBwebsitesBvirusBvirtueBvillageBurgeBuploadBunsubscribeB
ungratefulBunfunnyB
unbearableBultraBuhhhhBtrendyBtreasonBtraumaBtrappedBtornBtoesB
threatenedBthailandBthaiBthBterrificB	temporaryBteleportBtearingBtbBtalentsBswimB	suggestedB	succeededBsteerBspiritsBsoullessB
solidarityBshuttingBshoeBsheetsBshallowB	septemberBselfieBsalonBrocketsBrnbaBrisksBringsB
rightfullyB	returningB
respectingB	residentsBrepentB
registeredBredesignBrecognitionBrealisticallyBrangersBqBpurposefullyB
purposefulBpurchaseBpuppiesB	prospectsBprosBproposeBproduceBproblematicB
pleasantlyBplatinumB	platformsBpinchB
phenomenalBpersonalitiesBpeacefulB	partiallyBpantherB
overpricedBoverlapBoutletB
originallyB	operationBone’sBoldestB	occasionsBnzB	notoriousBnormsBnooBnoedBnicknameBnerveBnearbyBmrsBmortgageB
monogamousBmonitorB
moderationBmlmsB
minoritiesBmegaB
meditationBmangoBmagazineBlyricsBlvlBluxuryBlunaticB	lucrativeBlpBlawnBkissesBkinkBkicksBjewelryB
irrationalB
intriguingB	integrityB	instinctsBinfjBinconvenientBinadvertentlyB	immediateBicecreamBhotterBhostageBhoodBhipsBhighwayBhcBhawaiiBhauntingBharmedBhangoverBhailBhabitsBgrindingBglobalBgiftsBghostingBfuneralBfpsBforgivenessBfolkBfavsBfartsBfanbaseBfactualBexternalBextendedBexhibitBeuropeanBentertainedBengineerBengineBenemiesB	encounterB
electricalB
efficiencyBecstaticBdunkBduelBdoubtsBdorkBdncBdmcB	diversityB
distractedB	disregardBdisguiseB
developersB	describesBdarknessBcurlyBcuckedBcremeBcoveringB
counselingB
corruptionBcookedB	continuedBcontainsB	connectedBconciseB	competentB	clevelandBcirclesBchubbyBcheeringBcfB	censoringBcasuallyBcasinoBbuttonsB	butterflyBbulkB	brigadingBbrazilBbrawlBbrandsBbranchBbottlesBboomersBboilBbfvBbeliveBbassBbamaBbakeBbacksBathleticBatheistsBassumptionsBaspieBartsBarrowsBarrangementBarBangelsBaloofBalbumsBairedBacB9thB1980B😩B😏B☺BzoomedB
yourselvesBwweBwhooshedBwhereasBwaifuBvpnBvirginiaB	videogameBveganismBvanishedBunprofessionalBunderweightBultBtwistedBturtleBturnoverBtrickyBtraitsB
trainwreckBtouchesB	toothlessBthrilledBthoroughBthiefBthemedBthankingBtemplateBtargetedBsymbolBsweatB	suspectedBsuburbsBsubjectsBstudyingBstoresBstarringBstalkedB
snowflakesBsmackBsleeveBslaveBsjwBsisB	signatureBshudderBshieldBsheetBshakesBselflessBsecularB	secondaryBsealBseahawksBscariestBsaviorBsavingsB
sandwichesBrumbleBruledBrtBresignBresentBrepostsBrelyingBrelaxedBrefersBreferencingB	reductionBrealizesBraidersBquirkyBpunishB	primarilyBpresidentialB	preseasonBpredatorBpotterBposseB
positivityBposedBportBpoopsBpollsBpoleBpoisonBpmsBpipelineB
person’sBperkB
performingBpauseBpartisanBpanickedBpakistanBoverdoseBoutletsBoutdatedBounceBorgasmBoooohBoilersBofficesBnumerousBnukesBnotificationsBnormiesB	nicknamesBnephewB
negativelyB	necessityBnaggingB
mysteriousBmudBmountBmonstersBmoB
misspelledBmissionsBmightyBmensBmanagesBmaBlistensBliquorBlimitingBlesserB	legalizedBleapBleakingBlastsBladsBlacksBknockingBkindergartenBjvBjustificationBjuniorB	islandersBinvestigateB	introduceBinteractBinsertBinfamousBindulgeBincompetentBimplicationsBikeaBidioticBhostileBhornetsBhongBhmmmmmBheartbrokenBharryB
hahahahahaBguardsB	groceriesBgraphicsBgraderBgovBglueBglowBgenesBgainedBfundamentallyB	fortunateBformalBfleshBfeeBfeasibleB	fantasiesBfallacyBfakedBfaithfulBexploreBexplodeBexpectationBewB	ethnicityBethicalBemperorBelectBebayBdoinBdodgerBdistractionBdiscoveringB
disabilityBdipshitBdigitalB	deodorantBdemonBdeletingB
degenerateBdayzBcuterBculturesBctBcrushingBcricketBcrashingBcoyoteBcorporationB	copypastaB	convictedBcontrolsBcontributionBcontraryBconstructiveBcongratulateBconfuseBcomradesBcompeteBcloudBclientB	cinematicBchromeBchessBcheekBcheatersBcerealBcenteredBcelticsBcapsBcanesBcameoB
businessesBbronzeBbowlingBboreBbootyBbondBboiledBbelovedBbelieverBbareBbarberBbannerBayyyBavoidedBatheismB	assistantBassemblyBarkansasBappreciationBappalledB
apocalypseB
announcingB	alligatorBaliensBahemB	affectingBaffairsBadvancedBaddictB	activatedBabsentB	abhorrentB72B3xB32B216B1999B120B🤮B😬B😪B☹️B“thisBxpBxanaxBwomansBwinnipegB
windshieldBwikiBwidowBwiderBwickedBwhoresBwhineB	wellbeingBweakerB
watermelonBwatBwannabeBwackBvitaminBvillainsB	viewpointBvapingBurbanBup”BuntrueBunnecessarilyBuhhBtummyBtshirtBtrynaBtrunkBtravestyBtraumatizedBtisBtigerBtiesBthemselfBswedenBsweaterB
sunglassesBsubbedBstuffedBstudBstringBstoodBsquirrelBspyB	spotlightBsonicBsoldierBshitholeBshapedBshamBsettledBsessionBselfiesBseedBscrubsBscreensBscorpionBscopeBscentsB	scapegoatBscBsayinBsacredBrwooooshBrotationBrogueBrivalB	rewatchedB	rewardingBrevolveBreturnsBresumeBrestrictionsBrespondsBrereadBrelapseBrecycleBreboundBreactionaryBquadBpunishedBpubsB	promotingBprizesBprinceB	primariesBpresenceBpremiumBpremierBpreciseBpotatoesBpopeBpollBpolicingBpokingBplebBplagueBpilotsBpickyBpickleBpiBpeepBparrotBpairsBpainsB	painfullyBoverestimateBoutsBorientationB
oppressiveBolympicsBoctoberBoccasionB	obligatedBnyeBnumbBnuggetsB	nostalgiaBnopedBnooneB	newspaperBneBnaughtyBname’B	movementsBmobBmixingBmisuseBminionBmetroBmerrilyBmcdonald’sBmayoBmanufacturedBmannerBmaintenanceBlyricalB	lucasfilmBloadsB	liverpoolBliverBlibertyBlegalizeBleaguesBlbsBkickerBjerksBjerkingBishBinsultedBinsufferableB
infinitelyBindoctrinatedB	increasesBincestB	implementBhoardingBhilariouslyBheroinB
hereditaryBhelmetB
heartbreakBguardianB	greetingsBgrandmasBgrandfatherBgraceBgoddessBgeneticBgendersBgdpBgarlicBfunkBfukBfortunatelyBforestBfluidBfelonyBfaveBfathomBfalconsBfacepalmBextentB
expressingBethBessayBensureBendangeringBemblemB
elementaryBeditionBdyeBdsjBdrankBdragonsBdomesticBdohBdistinctionBdissB
dishonestyBdishearteningBdiseasesB
disconnectBdirtBdictatorB	developerBdemonstrateBdeliverB	definitlyBdedBdecreaseBdawgBcutsBcutieBcursiveBcurrencyBcrocBcreeperB
creativityBcoyotesBcontemptB	consumingBconstructionBconnectionsBcondomBcomaBcoconutBcoatBclosestBclawBcisBcigarBchunkBchuckBchemicalBchampionBchainsBcategoryBcapitolBcapitaBcanucksBcallerBcaliberBcaliBcafeBbratBbrahBbouncingBbopBbluntBblindlyBblastingBbillionaireBbettingBbanterBbabeBayeBawfullyBassistB
applicantsBapologizingBantibioticsB	anonymousBamerica’sBamazeB	algorithmBagenciesBadhdBachievementBaccountableBacademyB910B85B64B2kB2500B😣B😒B😀B💜B“soBzealandBzB
worthwhileBworkoutBwooshBwinnersBwigBwhomBweekendsBwaxBwarmsBwafflesBvoicedBviralBverseBvegansBvanBuntoBunhBunblockableBunbelievablyBunawareB
unansweredBubiBtxBtwilightBtrillionBtrialBtracerBtowingBtobaccoBtmBtissueBtiringBtimelessBthirstyB	thedonaldB	targetingBsyrupBswipeBswingsB	survivorsBsurfaceBsumBsuburbanBstupidlyB	spongebobBspectrumBspacesBsoyBsoilBsniffBsmallestBslursBslightBsleepyBsignalBshredBshatBsharksBsearchedBrustBrugbyBrocksBrobberyBritualBrewardsBretainBresultedB	repressedB	replacingB
remarkablyB	rejectionB	regulatedB
regrettingBregardedBrecommendedBrapistBpunctuationBpullsBpubBpsychB	proposingBprojectsBprogressivelyB
profoundlyBprofessionallyBproceedsBpoppingBpondB	poisoningBplayfulBpintBpickupBphdBpgBperformBpennyBpenguinsB
pedophilesBpearlBpeanutB	passengerBpartlyBpalmsBpackedBpacificBoverseasBoverreactingB	outsidersBoutcomesB	operatingBooofBnonstopB	nominatedBnodealBninjaBnationsBnationalismB	murderingBmooseBmonthlyBmodesB
moderatorsBmoanBmmmBmisunderstandBmisterBmisinformedBmeowBmemphisBmeatsBmcdsBmayorBmatrixBmatchupBmatchingBmarksBmarioBmanipulatedBmalesBloatheBlimitationsBlifelongBlickingBlemonsBlemonBlemmeBleggingsBleftyBleftiesBlandingB	knowitallBkfcBkdB	irritatedBintimateB	interfereBintentBinspireBinherentBingredientsBinfiniteBinexcusableB
inevitablyB
imprisonedB
illiterateBicyBhysteriaBhustleBhurrB	hospitalsBhookerB	honorableBhikeBhearsBharvardBhahahahahahaBgroupingBgrabbedB	goodnightBgoldfishBgemsBgateBgaslightingBfreshmanBfortBfoldBfmlBflippingBflewBfetusBfentanylBfeminineBfastestBfalloutBeyebrowBexposeB	explodingB
exclusivesB	evidentlyBeverythingsBestBepBentitlementBendorseB	endearingBeightBdvBdroolBdronesBdrivenBdrawnBdotBdloB	districtsBdistinguishBdisputeBdislikesBdishesBdingBdigitsB	dependentBdemoBdeclaresBdebitBdeadlineBdbBcuddlesBcruzBcrowdsBcringeworthyBcreationBcowsB	counselorBcooksB
contradictBconfrontB
confessionB	competingB	commentorB	collusionBcoffinBclimbBcleanerBcivilizationB	civiliansBchoirBchickensBchapterBchannelsB	champagneBcesspoolBcautiousBcatholicismBcatchesBcastleBcarolinaBcageBbrigadeBbreedingBboycottBbossesBbondingBboisBboBblunderBblitzBbleedingB	blasphemyBblandBbishopBbenzosBbeholdB	behaviorsBbeastsBbasicsBbankingBbangsBballoonBbaldBbafflesBback”B	backstoryBbabysBazulBazBayBawkwardnessBaugustBaudibleBattendB
attemptingBattachedBassertBaspectsBasinineBanyhowBanimatedBamaBallegationsBakaBahhhhhBafcBaccusesBaccomplishmentsB	abortionsB	abilitiesBaaaB2006B2002B🤷‍♀️B“you’reB“butB͡oByogaBwoooshBwooBwhewBwhaBweebBwarriorBwaBvprB	violationBversusBversionsBventingB	valentineBupgradedBupgradeBupdootBupcomingB	unlimitedBunisBunicornsB
underwaterBunbreakableBummmBuhhhBtsmBtropicalBtriggersBtribeBtriangleBtraitBtrainsBtougherBtollBtendiesBtenderBtekkenBtdsBtapBtameBtalibanBtaleBtackleBsymptomBswapBsustainableBsuckerB	substanceBsubhumanBstylesBstrokesB	storylineBstompingBstocksBstitchesBstickerBsteroidsBstatueB
starvationBstansB
staggeringBspiteBspawnBsockB	socializeBsmelledBsmashingBsmartestBsmackedBslotBslavesBslashBslappingBskippedBskiingBshrugBshowersB	shovelingBshillingBshapesBshaftedB
separationBscumbagsBsaverBsausageB	satiricalBsaintBs1BrushedB
rthedonaldB	rotationsBrotBroamBrimBrhymesB
rhetoricalBrhetoricBreworkBrewardedBrestrictBresortBrequirementsBrequestsB	repostingBremoteB	remainersB	relevanceBregimeB	redeemingB	recyclingB	recoveredBrankingBrabidBpuppetsBpugBpsychologistBproudlyBprotectsBprostitutesBproposedBpromisesB	professorBprobsBpresumeBponderB	playlistsBpimpleBpigsBphysiqueB	physicianB
philosophyBphilosophiesBphewBpetitionBperiodsBperiodicallyBpensBpedoBpedestriansBpatternsBparallelB	panderingBpacksBoverpaidBoutliveBoutfitsBopposeBoperateBonionsB	officialsBoceansBnudityBnraB
noticeableBnontoxicBnoisesBnobleBnintendoBnicestBnerdsBneedyBnavyBnasaBmutantBmultiplayerBmotivateBmoronicBmorningsBmississippiBminimalBmileBmeltBmedicationsBmdmaBmcBmayhemBmassageBmascotBman’sBmannersBmaneBmakerBmagnificentBmadnessBloggedBlnBlmaooooBliquidBlimbBlicensedBlibtardBlaserBladderB
judgmentalBjokinglyBit”BinvestmentsB	interruptB
intentionsB	installedBinfectedBindependentlyBimportB	immigrantBilBhowdBhoodieBhiveBhistoricallyB	hindsightBhighsBhighkeyBhesitantBhazyBhandicappedBhamsterBgumBgudBgrossedBgrinBgreensBgreeceBgrabsBgooseBgingerBgestureBgaBfundsBfuglyBfuckenBfraudBfortuneBformulaBfogBflirtBflightsBfirearmsBfianceB
fascinatedBfanboysB
fallaciousBextraordinaryBespBendgameB	endeavorsB	employersB
effortlessBefficientlyB	educatingBeagerBdynastyBdynamiteBdurrB
dominationBdominateBdodgyBdiverseBdittoB	discoveryB	digestiveBdictatorshipBdickedBdhBdeuceB
despicableBdesiresB	desirableBdesignerB	deservingBdenseBdefenceBdeclinedB	deceptiveBdarwinBdaredBdamningBcyclesBcutenessBcryptocurrencyBcrustBcrushedBcroppedB	cripplingB	criminalsBcrabBcornersBcontrastBcontactsB	constructB	consensusBconcentrateBcomplimentsB	complainsB
compassionBcolonialismBcoherentBcocaineBclosetedBclipsBclimbingB
classmatesBclarityB	civilizedBchurchsB	childfreeBchaptersBchantsB	championsBcfbBcbdBcampusesBcakesBbunnyBbuckBbrutallyBbronerBbridgesBbrideBbordersBboothBbmwBblazeBbigotsBbf4Bbf1BbentBbatsB	bartenderBbarstoolBbacteriaBassassinBarmourBarchB	arbitraryBappsBantisemiticBanthemB
animationsB	andromedaBanatomyBaltsBallyBallianceB	aestheticB	adulthoodB
adrenalineB	admirableBactressBactedB76B70sB62B52B46B3sB3amB37B21stB1kB🤷🏼‍♀️B🤷‍♂️B🤣🤣🤣B😳B😤B☹BzombiesByelledByardsBwrappedBwrBwimpBwifiB	whicheverBwelshB	weirdnessB	watermarkBwardrobeBvirginsBvintageBvineBviewedB	vegetableB	unwelcomeBunsatisfyingB	universalBunhingedBuncomfortablyBtweetingBtuitionBtrierBtremendouslyBtransphobicBtitanBtigersB	throwbackBthotsBteslaBtemporarilyBtakeawayBtailBsyrianBsworeBswitzerlandBsweetyBswB	surgeriesB	superheroBsuffersBsuckingBsubtletyBstunnedBstriveBstrawsBstrawBstinksBstatisticallyBstampsBstagedBstabBsprayBspotifyBsplitsBsplashBspinningBspikedBspideyB
speechlessB
specialistBsonyBsmokesBslidingBslaughteredB	slaughterBsketchBsipBsinisterB	singaporeBsincereB	simulatorBsimplerBsidedBshowcaseBshovelBshouldaB	shootingsBshhhhBsheeshBshavedBshakyBsellsBselfishnessBscytheBscummyBscrolledBscriptedBscandalBsadderBsacksBrockyB	rniceguysBrngBrivalryBreviveB	respectedBreporterBrenB	relievingB	releasingBrelativeB
regulationBrefundBreeducationB	reccomendBreachesBrapperBracingBquitsB	qualitiesB	purposelyB	purchasedB	punchlineBpunchesBpubertyB
prostituteB
proportionB
propertiesBproneBpromptB
productiveB	processesB	procedureB	principalBpricingBprescriptionBpremiseB
preferablyB	predictedB	predatorsBportlandBpopulismBpkBpinsBpinkyBpessimisticBperthBpersistenceB	performedBpeppersB	people”BpedanticBpeasantBpeanutsBpeachB
overweightBoverthinkingB	overnightBoverflowB
overeatingBoutlawBoutdoorBottawaBopticsBoilyBoffenceBofcBocdBobjectsBobesityBnwordBnukeBnpcBnieceBneutralBnervousnessBndBncaaBmuteBmottoB
motorcycleBmmmmmBmlbBmlBministerBmillionaireB
millenialsBmiBmfBmentionsBmeantimeBmasturbatingBmanlyBmanilaBmanicBmaniacBmanagingBmajesticBlodBlocalsBlightingBlightheartedBlibsBlibidoBlearnsBleansBlawsuitBlandlordBlandedBkneecapBkiwiBkittenBkeyboardBkeenBjuveBjuneBjournalBjBiudBipB	introvertBinstitutionsBinningsB	initiatedB	improvingBimplyBimmuneBimaBillegalsBidiocyBhungBhumiliationBhonourB	holocaustBhivBhintsBhikingBhassleBhairedBgunnaBgucciBgroundedBgreedBgoreB
goosebumpsBgofundmeBgoddamBgnarlyBgmsB	globalismBgirlyBgiggledBgiganticBgifsBgarageBfwiwBfussBfuneralsBframesBfordBfollowsBfoilBfleeingBfinlandBfillerBfightersBfifthBfieldsBfiatB
favouritesBfaultsB	fashionedBfartherB
fanfictionB	facetiousBezB	extremistB	extensionB	explosionB
exploitingBexplodedB	executiveB	excessiveBethnicB	escalatedBerrBerectionBequateBepitomeBenterBenslaveBenablingB
empatheticBeloquentBelfBearningB	dysphoriaBdukeBdrainingB
downloadedBdosesBdjBdivingB	disturbedBdisrespectedB	discourseBdiscountBdisagreeingBdisBdippedBdiamondBdiabeticB
developingB
devastatedBdestinationBdessertB
derogatoryBdemonsB	delusionsBdejaB
deflectingBdefiniteB	definetlyBdefeatedBdebutBdebacleBdealerBdeadlyBddrBdasBdangerouslyBdancesBcwBcurveB	crocodileBcringingBcredibleBcreatureBcraftedBcracksBcountersBcontrollersBcontractorsBconsultBconstitutionalB
consentingB	consciousBconfusesBconfrontationBconflictB
comparableBcommentatingBcluesBcitizenshipBcitationB	cigaretteBcheesyBcheckersBcharmBcbsBcarriageBcargoBcanonBcannabisB	cancerousBcampsBbustingBbummedBbuggedBbroccoliBbrinkBbrewBbrazenBbotwBborrowBboobBblurredBblizzardBblessingBblanketsBbisexualityBbelligerentBbelgiumBbeinBbeijingBbedsBbeamBbeaconBbananasBbakingB	audiobookBaudiblyBasthmaBashesBasfBartworkBarabiaBappearedB
apologisedB	apologiseBantsBantivaxBantBancapBalbertaBadvertisementBabroadBabnormalB8pmB700B55B49B2dB2amB28B22ndB2004B2001B1990B1500B110B105B100kB05B🤦🏻‍♀️B🙂B😱B👍🏻B🎶B❤️❤️❤️B≠B“weB“noB“i’mB
“don’tByumByoutubesByoghurtBwreckedBwormBwoofBwoodenBwonderfullyBwokenB	witnessesBwinrateBwholeheartedlyBwhiskeyBwavesBwashingBwakesBvoterBvoluntarilyBviolateBvidsBvestsBvergeBvenomBvaccinationsBuwuBunsafeBunionsB
unbalancedBtweetedBtunnelBtryinBtroubledBtrickingBtravelsB
transphobeBtransactionsBtraeBtownsBtowersBtourneyBtotesBtormentBtoriesBtopicsBtickleBthighBtfwBtextedBterminologyBtendencyBtedB	technicalBteamworkBteammateBteamingBtaxationBtaperBtanBtackyB
sweatshirtBswampB	suspicionBsupremeB
submissiveBstunBstreaksB
strawberryBstrainBstonerBstackBsryBsqueakBsquatBspontaneousB	speculateBspainBsoxB	somebodysBsoggyBsobbingBsnuggleBsnapsBsmokedBsmirkBslippedBslammedBskylineBskilledBsicknessBsheltersBshadingBsessionsBseriousnessBsequelsBsentientBseniorBsemesterBseekersBsecretsB	scratchedBscoopBsatanBsageBsacrificingBrubbingBruBrounderBrottingBropeBroastedBriflesBriamverysmartBrevokedBreunitedBretardsBresignedB
reputationB	reproduceBrejectsBregardsBrefugeeB
reevaluateBreconciliationBrebuildBrebelB
reassuringBreactedBrazorsBrandomsBrabbiB	quotationBqueueBpurseBpupilsB	publishedBpsychiatricBpsycheBprostateB
prosecutedB	pronounceBprogressionB	proactiveB	principleBpriestsB	pricelessBprestigeBpraisingBpraisesBpouredBpossessBposingBposesBpolygamyB
playgroundBplantsBplanesB
pittsburghBpitcherBpineBpillarsB
photographBpeskyBpermitBpdBpayoffBpawBparryB	paperworkBpanthersBpalletsBpalletBpackageBpaceBowlBoverreactionBosBorisaBorganizationsBoffsideBoffendsBoffenderBnyxlB
nonchalantB	nonbinaryBnewlyB	newcastleB	neglectedBnamingBmpsBmotivesB
monopoliesBmonoBmofoBmodelsBmobsBmillionairesB	migrainesBmigraineB
might’veBmethodsBmessagedBmergeBmeetupBmeetsB	mechanicsBmarryingBmadridBlurkingBlungBlunacyBlordsBloosingBloolBlonzoBloliBloldBlistingBlingoBlightlyBligaBlifesBleashBleaseBlearntBleafBlaziestBlayupBlayersBlawyersBlashesBkayakingBjudgmentBjfcBjellyBjarBit‘sBisoB	irritableBirresponsibleBirrationallyBinvasionBinternalBintermissionBintensifiesBinstaBinsignificantBinitialsBinflatedBinfantryBineptB	indonesiaBindifferentB	indicatorBindependenceBincorrectlyBimprovementsBimplementationBimperialBimaginationB	illnessesBillinoisB	identicalBiconicBiconB
hurricanesBhumidityBhookedBhoesBhitterBheartwarmingBhashBharmlessBhandingBhamstersBhalftimeBgrey’sBgraduateBgoofedBgooberBgoalpostBglBgeniusesBgdtBgaysB	gatheringBfuuuuckBfunctioningBfulfillBftsBfrogsBfreeingBfrayBforeseeableBfloodingBflavoursBfixesBfilesBfedsBfauxBfareBfakingBfairnessBfactoryBfacetimeB
fabricatedBextendBexposureB
explicitlyB	excludingBexcelBexceedinglyBeugenicsBetherBesportsB	equipmentBentpB	enlightenB	enjoymentB
engagementBemployerB	embracingBdunkingBdrumBdriedBdressedBdrasticallyBdolphinsBdodgesB
documentedBdocumentariesBdmsBdiscussionsB
disciplineB	disagreedBdignityB
difficultyBdiazepamBdialBdiabetesB	detectiveBdestructiveBdepositBdependBdeniedBdemandsBdeludedBdecBdamagingBdamagesBcustomizationBcupsBcrunchBcrossesBcrawlBcoversBcounterargumentBcoordinatorB
convincingBconsumptionB
conflictedBconcussionsBconcreteBconcedeB
commissionBcolumbusBcoloredBclothBclientsBcinemaBchristBchokingB	chiiiiikaBchicBcherryBcherishBchalkBchairsBchainsawB	centuriesBcentralizedBcensoredBcemeteryBccpBcavsBcaveBcaterBcasinosBcarrierBcannyBcableBbundleBbulletsBbrothB	broadcastBbricksBbrainsBbraggingBbournemouthBborkBbootlickingBboomerBbombedBbogglingBboatsBboardsBblowoutBbleachBblankBbjBbitchingB	birthdaysBbesideBbendsBbedtimeBbayernBbattlefieldBbatchesB
bargainingBbargainBballoonsBbakerB	backfiredBbachelorBavengersB	authenticBathleteB	associateBassignedB	assassinsB
artificialB	argentinaBampleBamongstBameenB
altogetherBallstarB	alcoholicB	aftermathBactivistBaccuracyBaccessedBabyssBabysmalBabominationB90dfB81B6amB51B3bB247B2021B2000sB🤗B😮B😞B😕B💪B🎂BzooBzonesBzeldaB	youtubersByousB	wrestlingBwrenchB	woman’sBwilponsBwieldBwhollBwhipBwhBwayyyyBwahBwackyBwaaaayBvocalB
vocabularyBvisitsBviewingB	viabilityBvetsBvestedB	verifyingBverbBvendorBurlBupstateBunsolicitedB	unnaturalBunlockedBunlockBunintentionallyBunimportantBuniB
unfollowedBuncontrollablyBunbiasedBtyrannyB	turnoversBtriviaBtrinityBtribesBtrekBtransphobiaBtransgenderBtrailersBtrailBtoughestBtossedBtorturedBtorbBtonightsBtomatoB	toleratedBthunderB	thumbnailBthreesBthrashBthibsB	theoristsBtheatersB	testimonyB	techniqueBtangentBtablesBtabBsyracuseB
sweetheartBsweeterBswearingBswayedB	sustainedB	suspendedB
supervisorBsumsBsummersB
subnauticaB
submissionBstrivingBstrictlyBstrictBstonesB	stockholmBstirsBstinkyBstingBstigmaBstewBstemB	startlingBstacheBsprayedBspoutingBspeedyBsooooooBsobBsnpBsnipeBsnappedBsmartlyBsmBslutBslurBsloganBslimeBslapsBslanderBslackB	sidewalksBsickoBshucksBshtBshrimpBshrekBshocksBshit”BshitlordBshiftingBshatterBseverityBscorerBscooterBscammedBsankBsandalsB	runescapeBrootsBrootedBromanBrollerBriskyB	rightwingBrichestBrhymeBrhapsodyBrevisionistBrevenueBreunionBretiringBrestedBrestartBresignationBreservedBrepublicBrepercussionsBrelicBrelatingBreeksBreceiptBrationalizeBramBracesBquickerBquadrigaBqldBpwbpdBpuzzledBpukingBpuddingB
psychologyBprudeB
protectingBprostitutionBprophetBprogB	profanityB	procreateB
processingBpriorityB
principlesBpriceyBprestigiousB
pressuringB
presentingBpresentationB	prematureBpoweredBpouringBpotusBpossibilitiesBposerBpolesBpoisonedBpoemB	plausibleB	plastiqueBplacingBpkkBpigeonsBphotoshoppedB
phenomenonB	pervasiveB
perpetuateB	perpetualBperpetratorBpensionBpenetrationB	paychecksBpathwayBpasswordBparticipateBparlorBparksBparadeBpanderBpancakesBpaleBpaintsBpacketB	packagingBoxygenB
overworkedBoutroBopiatesBongoingBolineBoklahomaBoffencesBoccursBoaklandBnukedBnuanceBnow”BnounBnotionBnoticesBnothinB	normalizeBnoodlesBnoodleBnolaBnobelBnbcBnarrowBn1BmumbleBmulticulturalB	mortifiedBmom’sB	mistakingBmisrepresentingBmidrangeBmereBmeetingsBmaximumBmatineeBmarylandBmarketsBmarginalBmainsBmagBlynchingBlynchBlustBlouisBlouderBlookoutB	locationsBlobbyBlmaooBlithiumBlionBlikewiseBlife’sBliberateBliableBlesBlatinBlargestB	labyrinthBkkkBkillinB
journalismBjointsBjointBjetBis”BiranBinvalidBinvadedB	intuitiveB	intrusiveBinsulinBinsistsB
insecurityB	injusticeB
initiativeBinfantBindianaBincompetenceB	imperfectBimmenseBimhoB
hyperbolicBhydratedBhumilityBhuggingBhowlingB	horseshitBhoorayBhooB
homophobiaBhominemBhomerBholderBhmmmmBhlBhitmanBhitlerBhidBheckinB	heartbeatBhbombBhatchBhallmarkBguttedBguidanceBgrumpyBgruesomeBgroundsBgriefingBgreaseBgradesBgovernorBgolfBgodspeedB	goddamnedBgoddamitBgluedBglitchedBgildedBgiftedBghBgeneralizationBgainsBgagsBftwBfringeBfrenzyBfratB	fracturedBfractureBfosterB	forgivingB	foresightBforeplayBforciblyBfocusingBfocusesBfloodBflickBflexingBflavourBflashingB	fireworksBfiendB	fictionalBfckBfattyBfandomBfalconBfairyBexposingB	explosiveBexploitativeBexpiredB	euphemismBermBenteringBenfpB
encouragedB
encountersB	emptinessBemotionlessBemojisBeliminationBeliBegBeditsBebtBeasterBdvaBdutiesBduoBduisBdsBdroneBdraggedBdoritosB	donationsB	documentsBdocumentingB
distraughtB
disclaimerBdiscardB	disbeliefB
dictionaryBdevotedBdesiredBdementiaBdeltaBdeflectB
definatelyBdbsBdameBc’monB	cultivateBcrybabyBcrushesB	crosspostB
criticisedBcrispyBcraveBcrampsB
craigslistB
courageousBcottageBcordBcopingB	contenderBconsumeBconsolesBconservatismBconnotationBconceptsBcomplicationsB
complianceBcompilationsBcominBcollabBclownsBclingingB	clickbaitB
classifiedBcheetosBchartsBchapBchantBchadBcentersBcarbBcannibalBcandiesBcamperB
camouflageBbulgeBbuffedBbromanceBbrakesBboundaryBboredomBboltonBboltBbohemianBbo4BblurryBbisexualBbidB	beveragesBbetrayalBbengalsBbelfastBbeggarsBbearerBbdayBbatteryBbatchBbasingBbarrenBbargeBbarfBballerBbaldingBbahahaBbackboneBauthoritarianismBausBasmrBappropriationB
approachedB	apprehendBappendixB	apologistB	apartheidB
antisocialBantidepressantsB
annoyinglyB	ambitiousB	ambiguousBambienBaltcoinBalmightyB	allegedlyBaligningBaightB
affiliatedB	affectionBadvocateBadoptionBadoptingB	addressesB
activisionB
accountingB63B58B57B54B5000B30kB20thB125B🙌B😘B😑B😇B💙B🎵B“stopB“sorryB“ifByolkByessirBxoBwwiiBwronglyBwrathB	world’sB	worldwideB	workplaceBwompBwizardsBwittenBwishedBwindsBwindedBwifeyBwidelyBwhydBwhiskyBwhippingBwhensBweebsBweeBwebosBwarmerBwankersBwalkerBvulnerabilityBvolvoBvodkaB
violationsB	vengeanceBussrBunwatchableBunsubstantiatedBunpredictableBuniformB	unfoundedBundocumentedB	undertaleB
undeniablyB	uncheckedBturdsBtskBtruthfulBtrumpsBtroutBtropeBtroopsBtransportedBtransportationBtrainedBtraceBtowelBtoryB	tolerableB	today’sB	timelinesBtileBtickledBthugBthroatsBthrivingBthinningBthievesBthat😂BtextureBtexansB	testamentBtenseB
tendenciesBtenantBtemperaturesBtelegramBteasedBteachesBtbmB	tastelessB
synonymousBswtBswornBswingingBswellB	sweetenerBsweatsBswastikaBsupremacistsBsupernaturalBsubzeroBsuburbBsubstantiallyBsubmissionsBstylistB	struggledB	strangestBstoveBstompBstitchBstfuBstealthBstbxB
standpointBstairsBspottingBspitefulBspiceBspellsBspectacularBspecifyBspacingBsourBsolvingBsoakBsnobBsneakingBsnarkyBsmoothieBsmfhBsmearingBslickBsledBskirtBskaterBskateBsizesBsinkingB	signalingBsiegeBsidelineBshrugsBshroomsBshovingBshovelsBshoutoutBshoreBsfwBsequenceBseoulBsendsBsenatorBselfawarenessB
secondhandBscreenshottedB	screeningBscottB	schoolingBschadenfreudeBscarredBscarfB	satisfiedBsaluteBruthlessBrushingBrudyBroyalsBroyaleBrosesBromanianBrollercoasterBroleplayBroflBrockstarBrockingBrockinBrnyBrlyBriderBrhinoBrevisitBrespectfullyB	resistingBrepresentedBremixB	remembersBrelieverBreignB	regrettedB
regressiveBrefugeB	reelectedBreelB
redemptionBrecoversB	receiversB	rebrandedB
reasonablyBrcomedyheavenBrantsBrankingsBrallyBragsB	radiationBradarBquotedB
questionedBqueBpushbackBpurgeBpumpsBpulpB
provincialB	proposalsBproposalBpropelBpronouncingBpromotesB	prominentBprofoundB	processedBpriestBpressesBprequelsB	practicedBpovBpooBpolitelyBpoemsBpnwBpnrBpleadingBpizzasBpilgrimB	picturingBphantomBpetrolB
permissionBperksB
perfectionBpelicansBpeckerBpdfsBpaywallBpaycheckBpatternBpatioBpassagesBparoleBparkourB
paragraphsBparachutingBpalaceBpairedBpadsBpackagesBp3BoyBownsB	overlordsBoutlierBoriginBoptimalBopenerBoooB	oligarchsBold”BoiBobsoleteBobsceneBobjBnuggetBnosesBnormieB	nonprofitBnonononoyesB
newspapersBneuroticBneedleBnearestBnatalBnarcissistsB
narcissistB	mythologyBmutuallyBmurdererBmultiBmourningBmosquitoBmonstrosityBmoneysBmonetaryBmocksBmmoB	misswiredBmissileBmisrememberingB	misguidedBminusBminionsBmermaidBmeasuresB
mayonnaiseBmattressBmatchmakingBmatchedBmansionB
mannequinsBmaniacsB	mandatoryBmakersBmaddenBmacsBlvBlustfulBlovinglyBlovejoyBlovableBlousyBlosBlongtermBloB
litchfieldBliftingBliftedBleftoverBlbBlabeledB	kinkshameBkimonoBkilljoyBkidnapBkiddosBkettleBkawhiBkabhiBjushBjumperBjokBjojoBjailorBjadedBjackpotB	irritatesBironmanBiraBinvolvementB
investmentBinvestigationsBinvasiveBintroductionBinterruptedBinstitutionalizedBinstitutionBinsiderB
injectionsBinhumaneBinfiltratingB
infidelityBinfatuationBindictedBincomingBimpracticalBimplicationBimperialistBimpactsBimgurBifunnyBifthanBickBhyperBhygieneB	hurricaneBhurrahBhugelyBhostingBhormoneBhooverBhookupB	homophobeBhimherB	hierarchyBhereticBhebrewBheavensBheaterBhavenBhashtagsBharmingBhandgunsBguy’sBgummyBgrizzlyBgrittyBgrannyBgrandsonBgrainBgotchuBgossipBgodlikeBgloveBglasgowBglanceBgksBghostsBghosteryBgelBgaugeBgangsterBgambleBfuzzyB
furloughedBfumblesBftBfryBfrighteninglyB	frequencyB
franchisesBfortsBfortressBfollowupBfloatBflexibilityB	flashbackBflamedBflameBflairsBfiscalBfireflyBfinanceBfiltersBfilmedBfiledBfilBfifteenBfierceBfeBfartingBfanmadeBfailuresBexpiringBexpelledBexmoBexaggeratingBetsyBescapingBescalateBequityBenviousBentryBentitiesBemotesBembarassingBelse’sBelementsB
electronicB	electoralBelderBdyslexiaBdysfunctionBdwellBdvdBduressBdupageBdryingBdrunkenBdriftBdrewBdpsBdowntimeB
douchebagsBdoublyBdonkeyBdonatingBdodgingBdocsBdmtBditchingBdisturbB	disordersBdiscriminatedB
discourageBdisconcertingBdisappointsBdisagreementBdilemmaBdifferBdictateBdfsB
determinesB	determineBdetainedB	designersBdesertBdeprivedBdeliversBdelightfullyBdecapitatedBdebunkedB	debatableBdaycareBdarkerB	daredevilBdankBdancersBdanBdamagedBdaftBcurlsBcropsBcrookedBcritiqueBcriticsB
criticizedBcraziestBcrassBcpsBcp3BcoupsB	cosplayerBcoolerBconvictionsBconventionallyBconvenienceBcontinuouslyBcontinuallyBconstructedB	considersBconsequenceBconsciouslyB
conflatingBcondoBconditioningBconcertsB
companionsB	commenterBclippedB
clinicallyBcliffBcitingBcircusBcirclejerkingBciaBchopsBcheatsBcharismaBcharacteristicBchampsBcasualsBcarelessB	carefullyBcanaryBcalendarB	bystanderBbuttholeBburritoBburnerBburglarBbufferBbuffaloBbrigadedBbrigBbriefBbrazzersBbranchesB	braindeadBbracesBbowlsBbotoxBbooingBbonkersBbolBbobBbmiBbluesBblocksBblindersBblastedBblackoutBbisonBbinsBbingBbicycleBbiasesBbewareB	betrayingBbenBbeirtBbeigeBbeggedBbaskB
bartendersBbarnBbankruptBbandaidBbaffledBbadgeB	backpedalBbacklashBbabesBauthorsB	astrologyB	aspergersBasgBasainBarntBarguedBarchitectureBarabianBantisB	anticheatB	announcerB
annihilateBangrilyBaneurysmBamygdalaBamputeeBamnesiaB
alligatorsBalliesBallergyBalikeB
alcoholicsBaksBairpodsBahahahaBaggravatingBagB	advocatesBadvisedBadjustmentsBadjustBacknowledgingBachievedBacdcBaccountabilityB
accidentalBacceptsBabramBabortBaboardBaaaandB800B70’sB67B40sB3ptB30thB180B140kB100xB🤷🏻‍♂️B🤷B🤦🏼‍♀️B😭😭B😫B😥B😍😍😍B😋B🔥B💗B🎵🎶B✨B♥B”B“toB“howB“goodB“better”B–B	‍♂️BzipByessssByellsByeetByawningByawnBwrecksBwrapsBwooooooBwoohooB	women’sBwildlifeBwhowhatBwhoohBwhatnotBwhataburgerBwhackBweirdosBweightsBweepB	wanderingBwaiversBwaistBvucBvisualizationBvirtualBviolatedBviableBviBvermontBvelvetBvainBvaccinationBvaBurineBupstairsBupsetsB	untrainedB	unleashedBunintentionalB	underagedBultsB	ukrainianBufaBtylenolBturkeysBtubesBtruthersBtrustworthyBtroopB
travellingB	transportB
translatesBtpBtowedBtouristsBtouchyBtotalitarianBtortillaBtorchBtorahBtomBtokyoBtlcBtlBtitansBtiramisuBtiradeB	timeframeBtiddyBtickedBthrowerBthroneBthirtyBthinkersBthereofB
therapistsBthatiBthanksgivingB	testiclesBtempBtangledBtallerBtaintedB	survivingBsurfB
supplementBsuccinctB
subsidizedBsubscriptionBstrikingB	stretchesB	stretchedBstreamerB	straightsBstoicismBstepmomBsteadyBstatismBstarkB	squidwardBsprinkleBspittingBspinoffBspinachBspermBspearBspeakersBspartanBsomoneB
somethingsB	solutionsBsolaceBsodasBsobrietyBsnowyBsnotBsnootBsnipersBsnatchB	smoothiesBsmokerB	smackdownBsinksB
simplisticBsigningsBsightingBshredsBshithouseryBshenanigansB	shatteredB	shadowbanBsequiturBsensationalistBsensB	semanticsBselfrighteousBselfproclaimedB	selectiveBseekerB	secretaryBscreamedBscholarsB	schedulesB	scheduledBscentB	sanctionsBsamsungBsadisticB
sacramentoBsabotageBsabatonBrttsBrtreesB	routinelyBroundaboutsBrottenBrookiesBrodeoB	robocoughB	robberiesB	riverdaleBripoffBrightfulBrichmondB
rewatchingBrevoltBreversedB
repetitiveBreopenBremovalBrematchB	remainingBreliveBreflexBredditsBrecourseB
rebuildingBreapBrdr2BrconspiracyBrayBrapesBrampageBrampBrallBrakeBquranBqueerBpuzzleBpurgedBpukedBptaBpsychsBpsychopathsBpsychopathicBpsaBprovidesB
protestersB
prosecutorBpropB
pronouncedBpromotedBpromBprollyBprogressivesBprofilesBpricedBprezBpreventsB	prevalentBpretzelB	pressuredB	powerplayBpowderBpourB
positionalB	portrayedB
popularityBpolarBpolBpmingBplugsBployBplaylistBplasticsBplankB	pineappleBpinBpicturedBphotographyBpepsiBpedalBpayrollBpawsBpausedBpaulBpastorBpassportBparticipatingBparentalB	paralysisB	panickingBpancakeB	palestineBpacketsB	overthinkBoverreactedBoverdueBoscarsBorgsBorganizeBop’sBoptBoppressB	oppositesBonwardBonethirdBoliverBoffsetBocsBobtainB
obsessionsBobsBnwBnvmBnutritionalBnursingBnovoBnourishBnostrilsBnorwayBnitpickBnerfedB	negotiateBnatsBnaeBmuntBmunchkinBmumsBmournBmotivationalBmotionBmorbidlyBmondaysBmoldB	moderatorBmkxBmisusingBmisstepBmisconceptionsBmindfulBmigrantBmetooBmeritocracyBmenaceBmeltedBmeleeB
megathreadB	megaphoneBmedievalBmedicareBmedalBmechBmaternalBmatchupsB	masculineBmarxistBmarxismBmappingBmanchildBmanbijBmallsBmaliceBlurkerBlungsBlulBloweredBlooselyBloganBlockupBlobsterBloathingBlivinBlipstickB
liberalismB	leftoversBleakBldBlawbreakersBlastingBlasagnaBlaopBlactoseBlabelledBkraftBkojiB	knowinglyBkid’sBjoysBjokicBjelloBjanitorBjammingBjagsB	isolationBireBinvadeBintjsB
intimidateB	intersectBinterracialB
internshipBinterchangeBinstructionsBinsightsB	innocenceBinitiateBinflammatoryBinfiltratedBinferiorityBindieB
indicativeBincB	impulsiveBimportantlyB	impatientBimmunityBimbecileBidgafBicelandB	hystericsBhungoverBhumorousBhumiliatingBhubbyBhubBhowdyBhoveringBhotelsBhostsBhorrorsBhornBhoppingBhipstersBhillsBheightsBheathensBhauntBharvestBhardhittingB
hardboiledBhandoutsB	halloweenBhahhaBgurlBgunfightBgsBgrossesBgrillBgratsBgrandchildrenBgramBgothBgoodlookingBgonerBglintBgitBgigglesBgenjiBgenevaBgenerateBgauntletBgangsBgaggingBfundamentalBfucktonBfuckkkkBfrozeBframedBfortyBformsBformedBforgettableBfootballersBflorenceBflippedBfleshyBfleetingBflatsBflaggedBflaccidBfishyBfibreBfemalesBfelonBfcBfastingBfarewellBfaqB
facilitiesBfabBexposesBexistentBexasperatedBewwBeventualBethereumBessaysB
escalationBerrrBentiretyBenlighteningBengagingB	enforcingBendureB
endangeredB	empoweredB	emphasizeB	empathizeB
embarassedBelitesB	egregiousBeffortlesslyBeffinBecologyB
dysmorphiaBdwBdustyBdunBdummyBduiBdubBdroidBdripBdrasticBdraftsBdownsideBdoubtingBdoubledBdopamineBdooBdominionBdndBdizzyB	divorcingBdivesBdistributionB
dislocatedB
disfiguredBdisdainB
discomfortB	disbandedBdippingBdipoBdigestB
diabolicalBdevoidBdevicesBdespairBdepartmentsBdenominationBdemoralizingBdemandedB
deliberateBdegradationBdeedsBdecryBdecentlyBdashcamBdankestBdaddysBdabsBcusB
culturallyBcuffsBcrustyB
criticismsB	cringiestBcreekBcozyBcourtesyBcopiesBcoopBconvictB	convertedBcontradictsB	contestedB
contendersB	containerBconsumedB	construedB
consensualBconnoisseurB
confrontedBconflictingBcompetitorsBcommunicatingBcommunicatesBcommonlyBcomedianB	colonizesB
collectingB
collapsingBcodesBclumsyBclosetB	clipboardB	classroomBcirculationBchungusBchuckingBchiBchewingB
cheesecakeBchaoticBcattleB	catharticBcateredBcatagoryBcastingBcartedBcarriesBcarltonB	caribbeanB	caressingB	cardinalsBcapeBcanoeBcagesBcaffeineBbuyerBburglarsBbunsBbunBbuggingBbucsBbruisedBbrothaBbroomBbreedersB	breakdownB	brazilianBbrandedB	brainwashBboogieBboilingB
blindfoldsBbitingBbipodBbillionairesBbikerBbiblicalBbenzoBbenghaziBbeanBbansBbanquetBballroomBbalconyBbagelBbackseatB	awkwardlyBawardedB
autonomousBauthoritiesBauldB
attainableB
attachmentBattBathensBassociationBarseholeBarsedBarousedBarghB
apologizedBapesBapeB	ancestorsBamasBalltimeBalignBalexaB
alcoholismBahlBahahBagonyBafaikBadvisorsBadventurousBadmiredBadjustedB
addressingB
acceptanceBaaaaandBa50B91B90’sB900B5xB4amB4000B25thB200kB1v1B1996B12thB01B🤷🏻‍♀️B😭😭😭B😜B😎😎B👏👏B“whyB“waitB“that’sB“stayB“snowflake”B“right”B“nowB
“name”B“myB	“lol”B	“littleB“fuckBzoomBzippersByelpByear”Byear’sByankeeBxxBwyomingBwtafBwoundedB	worcesterBwitcherBwitBwirelessB	willpowerBwikihowBwhistleBwhipsBwhippedBwhaaaatBwestminsterB	wendy’sBweightedBwebpagesBweaselsBwcjBwavyBwateringBwateredBwarmthBwarlordBwaitressBwaffleB	voluntaryBvistaBvisionsB	violentlyBviniB	vigilanteB
viewpointsBvibratorBverbotenBventiBveinBvegetariansB
vegetarianBvastlyBvaryBvacuumsButopianBuprightsBupheldBupfrontB	untreatedBuntouchableB
untalentedBunsolvedBunsatisfiedB
unreliableBunpackB	unlockingBunknownB	unhelpfulBundoubtedlyBunderstatementB	underpaidB
undercoverB
unculturedBuncoolBunconvincingBunconstitutionalBunbornBunarmedBujBtyrantsBtwineBtutorialBtuningBtumourBtuckBtruthsBtrumpetBtroughBtrolleyBtrolledBtroglodytesBtrimmingBtrenchesBtreadBtrapsBtraitorsBtraffickingBtrackedBtowBtovBtoreBtoadBtkachukBtitledBtimsBtimelyBtimedBtiktokBtigresBtightenBtidesBthunkBthugsBthrB	this’llBthighsBthielenBthiccBthem”BthemesBtexanBtestsBtestersB	terroristBtempsBteddyBteaseBtearedBtauntBtantrumsBtakeoverBtaintBtahitiBtacticalBtacticBswissBswipedBswiftBsweetsBsuspensionsBsuspectsBsurroundBsurgeB
supermodelBsunkBsundaeB	submittedBstumbleBstuffsBstringerBstressanxietyBstonersB
sterilizedBstereotypesBsteamedBstatistBstadiumsBstabbingBsrslyBsqueezedBspoonsB	sponsoredBspinsBspillBspeedingBspectateBsowingB	soulmatesBsosBsomewayBsolubleB
sociopathsBsocietysBsociableBsnowingBsnekB	snapchatsBsmiteBsmearBslumpBslothsBslipBslackingBskyboundBskitBskelthBskatingBskBsinnerBsingularityB	shutdownsBshotgunsB	shitheadsBshitbagBshinysBshiftedBshhhhhBshhhBshaftBsewingBsettlersBsettingsB	sephardicBseperateBsentimentalB	senselessBsenatorsB	semblanceB
selfesteemBseizureB	scriptureBscriptsBscrimsB	scratchesBscrapsBscootersBscientologyB
schedulingBscaringBscarierB	scarecrowBscammerBsavvyBsatsBsalsaBsalesmenBsailedBsagaBsaddleBsaberBs3BrunnerBrumourBruggedBrubbedBrpgsBroughlyBrogerBrmurderedbywordsBrlmBrigBriamverybadassB
rhelperbotBrfaBrexBretweetsBrestrainingBresourcefulB	requestedBrepresentingBrepresentativesB	renewableBrenaissanceBremovesBremorseBremakesBreliesBrejuvenatedBredsBrecommendationsBreaderBrcausalchildabuseBrationalistBrammedB	ramblingsBramblesBragingBraccoonBr1BqueuingBquantityB	qualifierBqpBqmBpythonsBpyramidBpushersBpurgingB	purchasesBpsnBps3B
prosperousBprosecutionBpronounB
projectileB	profitingBproceedBprisonerBprinterBprimalB
presumablyBpressedBpreschoolersBpredictionsB	precisionBppdBpoursBpositioningB	portrayalBpornoBpoppyBpoorestB	polyamoryBpolioBpointyBpoeticBpoesB
plummetingBplumbingBplatoonBplatonicBplasmaBpittsburghsBpistolBpianoBphilliesBpfftBpffBpfBpewBpervertBperspectivesB	personnelB
perceptionBpercentagesBpenguinBpeeledBpearBpatreonBpatchesBpasturesBpartyingBparasiteBparadoxB	pansexualBpansBpanelsB	pamphletsB	paintingsBpacBoxysB	overshareBoverheadB	outspokenBouterBouncesBosuB
originatedB	originalsBorganBorangesBoptimistB
oppressingBonsetBonesidedBomeletteBokcupidBokcBokay”BoffsidesBoffsB	offendingB	offendersBoccupyBobtuseBobjectifyingBoatmealBnxtBnutjobBnovaBnotingBnotifyBnotebookBnonreligiousB	nononoyesBnonchalantlyB
nominationBnmomBnmBnkBnippleBnerdyBnepotismB	negativesB
neckbeardsBnationalistsB
narcissismBnaivetyBmutedBmusterBmuricansBmurdersB	murderousBmunchakBmulletBmuddyBmtBmotiveBmorelosBmoodyBmontanaBmonopolyB	monologueB	mongeringBmonacoBmolestedBmoidsBmoddedBmochaBmnfBmnBmmBmk11BmixupsBmixerBmistaB	microsoftBmethodologyB	messagingBmenusBmenacingB	meltdownsB
mattressesBmathsBmasturbationBmassachusettsBmailmanBmadisonB	maddeningBlwBlumpedBlulzBlularoeBlottaBlooolBlogsB	logicallyBlmaoooBlivepdB
litterallyBlitterBlidBlickerBlibtardsBlethalBlecturesBldsBlaxBlaurelBlapseBlankyBlanceBlakesBlagB
lacklusterBlabsBknobBkittiesBkickersBkentuckyBkcBkatzBjuulsB	justifiesBjuryBjuniorsBjungleBjudgementalBjsutBjsBjournalisticBjohnsBjesusBjerkoffBjediBjarringBjailedBi‘mBitthisBinvestigatorBinvadingB
intestinesBinterviewerB
interstateBintersectionBintendsBintegrationBintegralBintactBinstallB	insomniacBinsomniaB	insidiousB	inshallahBinjectBinhalingBinfestedBinfernalBinfantsBinducingBinducedB
indecisiveB
increasingB
incoherentBimpulseB	improviseB
improperlyB
impossiblyBimportedBimmidiatelyB
immaturityBiiB	ideologysBideologicallyBideallyBiasipBhurdleBhumidBhospitalizedB	horsebackBhopelessBhonoraryBhollowedBhippiesBhim”BherosBhenryBheirBheaviesBheavierB	heartburnBheadbuttBhdBhawaiianB	harboringB
hamberdersBhallucinationsBhalifaxBhahhahaBguitarsB
guidelinesBgroanBgritBgreenerBgratingBgrandfathersB
graduationBgotoBgonBgoldsBgntBglucoseBglossedBglockBglobesBgeologyBgenitalsBgeneticsBgatherBgarnerBgardensBgardenBgamblingBfulfillmentBfuckkkkkBfrownBfrickingB	formulateBformingB	formationBforksBforgoingB	forbiddenB	footprintBflushedBflownBflourBflossBflippantBflinchedBfleshmatchingBflawlessBflBfknBfiresBfinnishB	finalizedBfillersBfeudB	fertilizeBfeintBfeigningBfedoraBfebBfavouredBfavourBfatalBfashBfarthestBfarmingBfarceBfamiliarityBfamilarBfakesBfaintBfacilityBfacelessBeyelidBeyeballBextraditionBexterminationB	exploitedBexpectsBexitedBexemptBexecB
exceptionsBexamsBevolutionaryB	evaporateBevangelicalBeuroBeurekaBescortedBerectBep4BenvoyB
entrapmentBenthusiasticB
entertainsBenhancementsBendorsedBendingsBenactBemoBemigrateB	embarrassBelkBeldersBegosBediblesBeczemaBeatinBeaseBdynamicsBdwellersBdupontBdummiesB	dumbassesBduckingBdrummersBdrownsBdrawsBdramasBdo”B	downrightB	downgradeBdourBdosayBdoorbellBdookuBdoggosBdoeBdistractionsB	distortedBdistasteBdissapointedBdisplayBdisappointmentsBdirkB	dinosaursBdimeBdildosB	digitallyBdieselBdictatorshipsB	dictatorsBdevoteBdetrimentalBdeterminationBdeterioratingB
detectivesBdestructBderpyBderangementBdepravedBdepictedBdemotedB
demisexualBdemiseBdementedB	demeaningB
deliveringBdegeneratesB	deformityB
definitiveBdefiningBdedmonBdebatesBdeanB	deadliftsBdeadliftBdayumBdaytimeBdaydreamBdashingBdanganronpaBdandyBdaeBdacaBcyborgBcrystalBcrustsBcrudeBcrowderBcrowBcrosspostedBcroppingB	crochetedBcritBcrispB
criminallyB	crackheadBcrackersBcrackerBcozBcouponB
countrymenBcountedB	countdownBcosplayBcorkBcopyingBcopiedBcopaBcooperationBcookieBcontradictedBcontraceptionBconstitutesBconspiraciesBconfoundBconfettiB
confessingBconeB	conductorBconcentrationB	concealerBconcacafBcompromisedBcomplimentaryB
complicityB
complexityBcompassionateB
commentersB	comediansBcombineBcolonyB	collapsedBcolderBcoincidentalBcoffeesBclichéBclearestBcivBcircumstanceBciderBchurchesBchoosesBchattyBcharacteristicsBcharBchantingBcertificateBceremonyBcentrismBcensorBcelebrationBcelebratingBcaveatB
categoriesBcastersBcashierBcashewsBcarvingBcarpetBcapturedBcaptureBcaptivatingBcannibalismBcandleBcampaigningBcamoB	cameramanBcactusBcackledBbwB	buzzwordsBbusinessmanBbushesBbushBbuildsBbuffsBbuffetBbrushingBbrowniesB	brotherlyB	brightestBbribeBbrianB
brexiteersBbreezeBbreastfeedingBbreakersBboxerBbowelBbourbonBborrowedBbopsBbooedB	bombshellB	bojanglesBbloodthirstyBblokeBblockadeBbloatedB	blindnessBbladesBbittersweetBbitchesBbikesBbetchaBbestestBbelongedBbelgianBbeepBbedsideBbeachesBbdsBbastionBbangedBbaneB	bandwagonBbaitingBbafflingB	badgeringBbaddiesB
babysitterBayyBawardsBawaitsBautonomyBaudacityB
attendanceB
atmosphereBatlantaB	astroturfBastonishingBassuredBassureBassumesB
assessmentBasdBartisticBarcticBarcadeBaquariumBapplicationsB	applaudedBappetiteBappearancesBapolloBaoBanzacBanticipationBanticipatingBanticipatedB	annoyanceBannouncementBanklesBanglesB	amusementB	alongsideBalmondBalliterationB	alienatedBajaxBaidBaflwBaffirmationBadvisorBadvicesBadvertsBadobeBaddidasBacuteBaclBacknowledgesBacknowledgedBacheB
accustomedBaccordinglyB
accidentlyBacademiaBabuserBaaandBaaaaaandB9gagB80’sB78B666B65B61B60sB5kB35kB29thB27thB240B23rdsB2030B1987B1623B13thB1210B10xB10thB10sB10mgB1000000B09B07B🤦B🙌🏻B😃B💯B
“your”B“yourB“whoB
“welcomeB“toxicB“justB	“i’llB	“gamersB“areBzenByoutuberByouthsByogurtByiuByankByaaaaaasBxoxoBwrongiB
wrongdoingBwretchedBwreckingBwoooooBwon‘tBwizardryBwittleBwishlistBwiredBwipingBwingerBwilBwigginsBwidthBwhoopinBwhodBwhistlesBwhalesBwesBweptBwelthamBwelderBweirdsBweighsBwehoBweaselBweakestBwcwBwatchmenBwardB	wallpaperBwalesBvoxBvitaminsBvirtuousBvinylBvinegarBvikingBvicinityBveyBverifyBverifiedBvaxxerBvaselineBvasB	variationB
vandalizedBvaluedB
validatingBvaginalB	vacationsBuselessnessBupdootsBunwantedBunsurprisingBunsuccessfulBunrepresentativeBunregulatedBunqualifiedBunpaidBunoBunlawfulBunityBunitsBunfathomablyBundoBuncertaintyBuncaringBuncannyBunapologeticBugaBtyingB	tutorialsBturfBturboBtumorBtrumpieBtruerBtrippedBtribalB
tremendousBtreatyB	treadmillBtreacherousB	traumaticBtransportingBtransformerB	transfersBtranniesB
traitorousBtrailsBtradesBtraderBtodoBtoasterB	timestampBtikiBthumpB
throwawaysBthriveBthrillsB	threadingBthosBthomasB	things”BtheeBthcBthaBtexturesBterroristicBtenureBtemperBtechnologicalBteabagsBtastingBtartBtarotBtappedBtapedBtannedBtailoredBtailingBtabsBtabloidBtabletB
systematicBsyncB	synagogueBswearsB	swastikasBswappedBsurvivesB	surpassedBsupriseB
suppressedB	supplyingBsuperiorityBsupBsumoBsuffocatingB
suffocatedB	successesB
substituteB	subjectedBsubconsciouslyBsuBstupiderB	stumblingBstudiedBstubbornB
structuredBstrikerB	strengthsBstopedBstoningBstoicBstockingBstockedBstickersBsterileBstereotypicalBstbxhBstassiBstarveBstaresBstainsBstagBssdB	squirmingBsquintBsquashedBsquadsBspringsBspousesBspookedBspontaneouslyBsponsorshipsBspinnersBspikeBspecialistsBspanB	spaghettiBspaB	sovereignB	someplaceBsoleBsolarBsoftwareBsocdemsB	snugglingBsnoopingBsneezingBsmurfBsmithBslumBslowedBslobBslappedB	slanderedBslammingBskimmedB
skateboardBsixteenBsirensBsirenBsimsB
simplicityB
similarityBsignificanceB	sidelinesBshrinkBshreddedBshoutsBshottyBshitheadBshippedBshinglesBshinesBshiftsBshieldsBshhBshamesBshamelesslyBshadesBsfvBservantsBsepticBsensorBsemitismBseizeBseinfeldBseedyBsecuredBscrewsBscreechBscissorsBscifiBschemesB	scepticalBscarcityBsavorBsapB
salmonellaBrveganBrustyBrubsBrthathappenedB	royaltiesBrostersBropesBrodeBroastsBrnflBrnewsBrmBrkanyeBrivalsBriteBrisingBriotsB	ridiculedBriBrevolutionaryB	reviewingBretainsB	restrictsBrestoreBrestingBresearchingBrequiemB	reportersBreplaysBrepealBrepairB	renditionB
remasteredBremainedBreloadB
relentlessBreichBregulationsBrefreshBreducingBredheadB	recruiterB
recommendsBrecommendingB	recessionB	receptorsBrebootB
reappearedBrbitcoinBrbestofBrazerBrappersBrantingBraleighB
rainforestBrailingBraiderBragnarokBrackedBraBr2B	qualifiesBqualificationsBqaBputtinBpushesBpusherBpursuitB	purgatoryBpuntBpunsBpubesB	psychoticBpsychosBpsstBpsgBprotocolB	protectorB
prostheticB
progressesBprogrammersBprintingBprintedB	prettiestBprettierBpreserveB
predictionB	predatoryBpredatesB	precedentBpowellBpoultryB
postmodernBpossumB
positivelyB	portraitsBpoppinBpontiacB	pollutionBpollutedBpodiumBpoachedBpledgeBplaytimeBplayableBplatBpixelBpiercingBpetulantBpetiteBperpetuatingB	permittedBpermBpegB
pedestrianBpebbleBpeakyB
peacefullyBpdfBpatrolBpatentlyBpastryB	passportsBpassionatelyBpartingBparaphrasingB	paralyzedBparadiseBpalsyBpalsBpaletteBpajamasBpaddlesBpackingBozBovertonBovertlyB
overthetopBovertB
overstatedB	overreactBoverlookBoverlayB	overgrownBovationBoutbreakBottersBopioidBopiateBopeBonoffB	oneyplaysBoneselfBomB	oligarchyBofflineBodsBoctB
occurrenceBobvsB	obsessingBobsessB
objectivesB	objectionBobeyBnuttersBnutshellB	nutritionBnuggsBnudesBnotifBnoticingB	nostalgicBnosB
noooooooooBnoooBnoobsBnonsensicalBnominateBnihilismBniecesBneverthelessB
neoliberalBnegotiationsB
negotiatedBneedlessBnebraskaBnbBnatoBnapoliBmummyBmriBmouthyBmotivationsB
motivatingBmortarBmortalsBmontageBmom”BmolyBmoleculeBmodifiedBmodestBmodelingBmmmmmmBmistBmisspokeBmissouriB
misogynistBmisinterpretedB
misfortuneBminingBminerBmindlessBmillsB	migrationBmetricsBmermaidsBmerchBmercedesBmentholBmemerBmeltingB
mediocrityB	mechanismBmaysBmathematicallyBmassesBmartyrBmarinateBmarginBmanuallyBmantraBmantisB
malevolentBmajorsBmailingB	magnitudeBmacheteBlyricBlunchesBlove”BloveableBlotionBloopsBloopholeBloooongBloneBlololBlipsyncBlinuxBlinemanBlimesB
likelihoodB	lifesaverBliceBleveledBlettuceBlesbianBlensBlegislatorsBleftleaningBleechesB	lecturingBlebatardBlawdBlaundryBlaunchedBlatheBlamestBlalBladdersBlabelingBkpopBknowjustBknobsBknickersBkneejerkBkinoBkikiBkidneyBkenpomB
justifyingBjusticesBjuicingBjuicesB
judgementsBjoinsBjimB	jewelleryBjerkedBjammedBjacketsBjaBitchingBistanbulBirreversibleBiraqBipadBinvolvesBinvitingBinventedB
invariablyBintrovertedBintimidatedB	interveneB	internetsBintercourseBinteractionsB	intensityBintendBintellectuallyBintegratingBinsufficientB
installingBinsistedBinsipidBinnitBinkBingameB	inflationBinfinityB
infallibleB
inebriatedB
indigenousB
incentivesBincaseBinbredBimposingBimpeachmentBimaxBiiiBignoresBidiomBidcBicuB
hypocritesB	hyperboleBhuskyBhushBhunsBhumaneBhpBhowlB	housewifeB	horseshoeBhome”BhoggingBhoaB
historiansBhippoBhiphopBhipaaBhinderedBhillsongBhighlightingBheyyyBhesitateBheritageBhelplessBhelmetsBhelmBhearsayBheapingBheapBhealerBheaderBhazmatBhaystackBhartfordBharrisBharmsBhardwareB	handshakeBhandlesBhaircutsBhabsBhabitatsBhabitatBgunshotBguestsBgrosslyBgrilledBgriffinBgrasBgrammaticalBgrabbingBgourmetBgottemBgospelBgoooooBgoinB	goddamnitBgloomyBglitchBglancedB	gladiatorBgiverBginBgigglingBghettoBgfaBgentsBgentlyB	gentlemenB	gentlemanB	generatedBgenderedBgarfieldBgardenerBfuuuckBfuturesBfunhausB
functionalBfrostyBfrightBfrecklesBfraudedBfptpBfoulingB
formattingBforgivenBforestsBforeseeBfoldingBfolderBfoidBfluorideBflooredBflockB
flashlightBflamerBfkinBfishedBfirBfingernailsBfillingBfigurativelyBfifyBfiddleBfetusesBfemshepB	femenistsBfeaturedBfearedB	favorableBfavorabilityBfaunaBfascinationBfarmerBfangirlsBfanboyB	factoriesBexportBexpoB	exploringBexploitB	expansiveBexpBexitingB	exercisesB	executingB	excusableBexclusivityB	exclusionBexaggerationBexaggeratedBevictionB
everlovingB
euthanasiaBetlBethicBerraticBerrandsBergoBerasedBepsilonBepsBepcotBenzymeBenvironmentsBentriesBentpsBentitlementsB
enthusiastBenthusedBensuingB
employmentBemphasisBemergenciesBelitistBedgesB	ecosystemBeatersBdystopiaBdysfunctionalBdyinBdwightBdupedBdunksBdungeonBdumpingBdulledBdualBdsaBdrunksBdrownedBdrivewayBdreamedBdreadedBdoxingBdownloadingBdownedBdoubtedBdosageBdoorwayBdon‘tBdonorBdongB
dominatingB	dominatedBdodoBdnpBdiyBdivertB
divergenceBdistastefulB
disruptiveBdisqualifiedB	disputingBdisproveB
dispensaryBdiscouragesBdisadvantagedBdirtiestBdireBdintBdildoB	dignifiedB	differingBdiffBdiaperB	diagnosesBdiagnoseBdevilishBdeusBdetractBdespisesB	despacitoB
descendantBderangedB
depreciateBdenounceBdemonicBdemographicsBdeletesBdegromB	degradingBdefyingBdeforestationBdefensesBdeemedBdeedB
decompressB	decliningBdecksBdecencyBdeceasedB
dealershipBdealersBdealbreakerBdasherBdartBcustomsBcurryBcurledBcurdsBcultsBcubesBcubeBcubbiesB	crybabiesBcrusadeBcruiseBcrowsBcrowdingBcrossbowBcrochetBcribsB	creepiestBcreatorsBcrateBcraftyBcovertBcouponsBcottonBcostumesBcosmosBcorrectsBcorrectionsB	cooperateBconveyB	continentBcontentiousBcontaminatedBconstraintsBconsciousnessBconsBconquerBconnecticutB
conformityBconditionerBconditionedBconcentratingBcomputeBcomprehensiveB
competenceBcompensationBcomparisonsB	commitingBcommentatorsB	comicallyB	comfortedBcoltBcoloringB	colombianB	collapsesBcocoBco2B	clutchingBclosesBclosemindedBclicksBclickingBclicheBcleanedBclashBcivicBcircaBcigarsBchromosomesBchilliB	childcareB	chihuahuaBchequesBcheddarBcheapestB
charitableBchaiBcerebralB
censorshipB
cellphonesBcatchyB	cardboardBcan‘tBcansB
cancellingBcamelBcalzoneBcalmestBcalBbunkBbunchaBbuilderBbuggyBbuffoonBbubblingBbsvBbrushedBbrowserBbrownieBbroadcastingBbritsBbrilliantlyBbrighterB
brightenedBbreweryBbrethrenBbreathtakingBbreastsBbreakoutB	braceletsBbpaBboxesBbotchedBboostedBboomingBbondsBbo3BblurayBbloomingBblondesBblmBblehBblamesB	blamelessB	blackmailBbirthedBbioshockBbioavailabilityBbf5BbetraysBbestiesB
beginningsB
beforethisBbeatableBbeastlyBbearingBbattlingBbatteredBbarleyBbarefootB
bankruptcyBbailedBbaggageBbadmouthingBbackpackBbabylonBbabblingBbaBb4BauntieB
attributedBatlanticBathletesBasstB
assistanceBaspiesB	askredditBashBasbestosBarsesBarrivesB	arrestingBarmoredBarbitrarilyBappropriatelyBappointmentsB
applicableBapplauseB	appallingB	apostolicB
apartmentsBantagonisticB	announcesBampedBamishBamericasBamendsB	allowanceB	alleviateBallenBalignedBalcsBakronBakinBaircraftB
agreementsBagingB
aggravatedBagain”B	afterlifeB
affordableBafarBadvertisementsB	advertiseB	adoptionsBadmitsBadmiringB
adjustmentB	adhominemBaddictsBadamantBaccordBaccommodateB	accessingBacabBabusesBabusersBabstractBabsenceB
abductionsB9pmB98B82B750B6kB50thB50sB450B420B3mB360B31stB230B19thB1984B1970sB170B1625B14thB1212B11thB🤪B🤩B🤦‍♂️B🤣🤣🤣🤣🤣B🤣😂B🤞B🤕B🤓B🙏🏽B😡B😝B😛B😂😂😂😂😂B💖B💓B👍🏼B🐃B🍰B♫B♡B“selfB“religionB“andB‘emB‘98BδB£10BzingerBzimBzerocalorieByuhByoungestByooooByheaB
yesterdaysByeppByeezusByearoldsByareB	yardstickByankingBwtaBwrestlemaniaBworshipsB	worldviewBwootBwooowBwlBwindyBwiBwhitesB
whitehouseBwhisperBwheredBwhat’dBwhatchuBwerkBweplayBwellwrittenB
wellingtonBweldingB	welcomingBwebcamsB
wealthiestBwd40BwayyyB	wayne’sBwaxingBwaspBwasamBwarnsBwardsBwanderBwanaBwallowBvpBvoodooBvoicingBvogueBvisuallyB	virtuallyBviolatesBvideB
victimlessBvicBverticalBverballyBvenusBventureBveggieBvaxxedB
variationsBvaporBvampiresBvalorBvaguelyBvaccinatingBupwardB	upgradingBuntimelyBunsubscribedBunsubbedBunsubBunprotectedBunofficiallyB	unmaskingBuniversitiesBuninterestingB
unfinishedBuneasyBundoneBunderwhelmingBunderprivilegedBunderappreciatedBuncharacteristicB	uibiteyouBuhgoodBudemyBublockBtylBtweaksBtweakingBtwdBtwasBtunaBttigersBttBtsnBtryedBtruckerBtruBtrippyBtrimBtrickleBtributesBtreyBtrenchBtraumatizingB
transplantBtransmissionB
trajectoryBtractorBto”BtouristBtourismBtoungeBtotalingBtossupB	tortillasBtorBtoppingsBtoo”BtommyBtomahawkBtokBtoddlersBtippingBtinnitusBtingleBtimBtikkaBtikBtightsBtifuBtidbitBthyB	thrillingB	threesomeBthisisB	thesaurusBthermBtheoristBtheologyB	theocracyBthemsBtheistBtfaB	terrifiesBteriyakiBtentB	tennesseeBtenancyBtemptingBtemptB
teleportedBtaxpayerBtaurusBtastefulBtaiwanBtailsBt2BsynonymB
sympathizeBswtorBswollenBswellingBswelledB
swallowingB
sustenanceB
suspensionBsurveillanceB	surrenderBsuppressingBsuppliesBsupervisionBsuperpowersBsuperfluousBsuperficialB
superbowlsBsunriseBsundaysBsummonedBsummitBsuitableB	suffocateBsufferedBsucceedsBsub”Bsub’sBsubscribingBstylishBstudiosBstrqB	strippersBstripperBstripedBstricterBstressesB
strategiesB	strategicBstrandedBstraightforwardBstorytellingBstoopBstipulationsBstiBstenchBstdsBstatistsBstardewBstampBstalkerBstainBstagesB	stabilityBsrirachaBsqueezeBspurredBspunBspreadsBsponsorsBspiralBspiderverseBspewingBspeechesBsparseBsparkedBsowBsovereigntyBsousBsourcedBsoundingBsoughtB	sophomoreBsophisticatedBsofterBsofaBsociopathicB
socialisedBsnuBsnortBsnitchBsniperBsniffingBsnakesBsnagBsnBsmoothlyBsmokinBslyBslowsBslomoBslipsBslinkyBslidersBslidBslicedBslavingB
skillfullyBskeletonBsixersBsitterBsithB
simulationBshutsBshutoutBshuffleB	shrinkingBshot”BshooedBshockerB	shirtlessBshelterBshariaBshanghaiBshaggyBshagBsg12BsewageBserumBsepsisB
separatingBsemichahB	selfworthBselfdeprecatingBselectBseduceBsecondlyBscubaBscreenshotsBscoutsBscousersBscorchedBscoopingBscientificallyBschlockBschizophrenicBscatBscamsBscammersBsc6BsavantsBsaugB	saturdaysBsangB	sanctuaryBsamuraiBsamplesBsalemBsalahBsaharaBsafestB
sacrificesB
sacrificedBs2gBs2Bs11BrussiasBrunwayBrundownBrumBrudenessBrtrashyB
rtitlegoreBrsoccerBrrareinsultsBroyBrowsBrowdyB	roommatesBrocketedBrmoviesBriledBrileBridiculousnessBridersBricherBrevolverBrevolutionariesB
revelationB	revealingBretroBretreatBretortBretoolBretardBresurrectionBresubmitBrestsBrestrictionB	restoringB	resonatesB	resistantB
resistanceBresetsBreservationsB
resentmentB	resentfulB
resemblingBrescuedBrentingBrenderBremedyBremarksBreliablyBreleasesBreinBregulateBregionalBrefreshinglyB
reflectiveBreeferB
reeeeeeeeeB	redundantBredistributionBredeemedB	rectifiedBrecreateB	recogniseBrecievedBrecessedB	receptiveB	receptionBrebuttalB	rebellionBreasonedBrealisesBreadersB	reactableB
rdankmemesBrchoosingbeggarsBrarityBrapidlyBramadanBrailsBrailBquieterB	quicknessBquickestBquarantinedBpurdueBpuntingBpunisherB	publicizeBptrBpsychologicallyBpsychicBprovinceB	providersB
proverbialBproverbBproudestBprotipBprotestsB
protectiveBprotagonistBpromiscuousBprolifeBproletariatB
project”B
prohibitedB
professionB	producingBprobableBprivacyBpriusBprintsB	primetimeB	primantisB	pretendedBprerequisiteBpremiumsBpremiereBpregananantBprefersBpreemptivelyBpredominantlyBpraxisBpraisedBprageruB
powerhouseBpostageBportugalBportrayB	pornstarsBporesB	populatedBpoppersBpoorerBpoofBponytailBpongBpollyBplumberBplowBpliersBpleasesBplayinBplaguedBpitchesBpishBpiratesBpipesBpintsB	pinkertonBpiggyBpiesBpierBphrasesBphotographerBphonyB	pewdiepieB	petrifiedBpenpalBpendingBpencilsBpeerBpeedBpedsB	pediatricBpedestalBpeBpdxBpcsBpbrBpbBpaypalBpayerBpaybackBpausingB
patriarchyBpassageBpartnershipBparticipantsBpartialBpardonsBpantBpandaB	palpatineBpalpableBpacedBozoneBoxBoverwhelmedBoverturnB	oversizedBoverpoweringB	overdriveBoutputB
outlandishB	other’sB	organisedBorganiseBoperaBoooopsBooohhhBoodlesBoncourtBomnibusBomlBoldfashionedBoldenBoffhandBofcourseBoctopusBoctaneB
occupationBobstacleBobsessivelyB	obsessiveB	observersB	observantBobgynBobesB	obamacareBoasisB	nutrientsBnutrientBnurseryBnurkBnugeBntBnparentsBnovelsBnotreBnostrilBnosleepBnoooooooBnontrumpB
nonstarterBnomBninoBnineyearoldB	nightclubBniemiBnickBnicheBnfccgBnexusBnexBnewsroomBnewestBneurotypicalBnepalBneoBnemoB
neighboursBneighbourhoodBneighBnegatesBnedBnecklaceBnebulousB
nauseatingB	nationalsB	narcoticsBnandosBnailingBnachosBn64Bmusic”B	musiciansBmushyBmushroomBmurmurB	murderersBmumblingBmuffinsBmsuBmphBmotorB	morrowindBmoroccoBmopBmoistBmoicanoBmockeryBmmosBmmaBmistypedBmistressBmisinterpretingBmiscarriageB
minimalistB	millionthBmillB	milestoneBmilanBmidwayBmidfieldBmicsB	microwaveBmichelinBmetropolitanB	metahumanBmeritsBmergedBmercuryBmerchandiseBmepsB	menstrualBmeltdownBmelodramaticBmelodiesBmeloBmeeeeBmeditateBmedicaidB
mechanicalB	measuringBmbtiBmatteBmateyBmasturbatedBmasteredBmascotsBmaryBmarkingBmarkersBmarineBmarchesBmanureBmanslaughterBmanipulatingBmaneuverBmammalBmamBmalaiseBmagsB
magnitudesB
madridistaBlyftBluvBlunaticsBlukewarmBlucioBlsuBloudlyBlooooolBloonyBloomingBlooBlongtimeBlonerBlolzBlolololBloathedBloadingB	lipsticksBlimpBlimewireBlilleBlikewhatBlightenBlibertarianismBlgdBlgBlexusBleverageB
legitimizeBlegalizationBlegalityB	legalisedBlatchBlasB	landslideBlampBkyBkrgBkrBkpinsBkotakuBkoolaidBknuckleB	knoxvilleBknotBkitchensBkirklandBkindleB
kidnappingBkiaBkhajitBkeemunBkansasBk9BjustnosBjurassicBjunkiesBjuniperoBjonesBjihadiBjestBjeepBjamesBiunnoBislaBiphonesBinterventionBintermittentBinteriorBinteractingB	intenselyBinstitutionalBinspirationalB	inspectedB	insinuateBinjureB	inheritedB	inflatingBinfieldBineffectualB
industriesBindignationBindepthBincomprehensibleB	incognitoBinboxBimposeBimplodeBimplicitBimplementingBimplausibleBimplantBimminentBiguanasBigniteBifsBickyBhystericallyBhurtyBhurtfulBhuntersBhunchBhumBhulkBhtownBhrsBhrcBhqBhottestBhortonsB	horrifiedBhornsBhootBhonesltyBhondaBhonBhomosexualsBhomosexualityB
homosexualB
homophobesBhomicideBhomelandBhofBhoeBhmbB	hispanicsBhiringBhiresBhillaryBhe‘sBherpesB	hellscapeBhellsBhecticBheathenBhealedBheadshotB	headachesBhazardBhauntedBhatchetBharaamB	haphazardBhangryBhalvesBhaloBhallucinationBhallsBhairsBhahahahahahahahahaBhahahahahahahaBhackedBgwentBgunstBguffawedBgrumpsBgrrBgroanedBgreetBgravesBgrapplerBgranddaughterB
grandchildBgraffitiB
governanceBgoodieBgoobleBgongBgoldmineBgodtierBgobbleB
goaltenderBgncBgnatsBglutenBglossBgloballyBglimpsesBgfsBgesturesBgenresBgeniousBgeneralizationsBgeneBgearsBgcBgaydarBgatoradeBgatlingBgaspingBgasolineBgaslightBgapsBganksBgalBfyreBfunkyBfumbleB
fulfillingBfuddBfrugalB
frightenedBfridaysBframingB	fragmentsBfractionBfounderBfouledBforsakenBforfeitsBforeskinBforemostBfodenBflungB
flatteringBflaskBflashyBfkingBfixableBfirst”BfingBfindingsBfinancesBfilteredBfillsBfiguringBfesterBferretsBfellowsBfellasBfeefeesBfearsomeBfearmongeringBfdrBfcukBfazeBfaxBfatlogicBfatalityBfashionableBfarrightB	fanaticalBfalselyBfactionsBf5BeyB
exreligionB	expressedBexplorationBexorcistBexistentialBexgirlfriendB
exemptionsBexacerbatedBevolvedB	evergreenBevaluateB	etiquetteBethicsBestrogenB
estimationBestateBesoBescortBerectedBepidemicBensuringBendorsementBembarrassinglyBelevenBelevatorBelementBelegantBegotisticalB	eggshellsB	effectingBeeBebBeastsideBearthersBdwellingB	duplicateBdunceBdublinBdubbedBdubaiBdtwBdtrBdruggieBdruggedBdribbleBdressesBdrainedBdownsBdoublethinkBdotaBdormBdootBdoggyB	dodgeballBdmgBdjangoB	displayedB
disneylandBdisgustsBdiscouragedB
disappearsBdisapointedBdisadvantagesB	directorsB	directingBdioxideBdinerB	diffusionB	diffrenceBdifferentialBdiaryBdiabloBdfacBdevoutBderpBderivedB	departureBdensityBdenominationsBdemoralizationBdemonstratedB	demonizedBdemeanBdelveBdelicateBdelaysBdefinedB
deficiencyBdeffoBdefenselessB	deceivingBdearlyBdeadpanBdazeBdauntingBdaringBdancedBdamnsBdamnnnnBdabbleBcutiesBcubanBcssBcrutchB
crosspostsBcritterBcretinBcreedBcredentialsBcreaseBcrawfishBcravingsBcravesBcrankingBcrabtreeBcowardlyB	cowardiceBcounterpointB
counselorsBcosyBcorpusBcornyB	cornbreadBcorinthiansBcoppersB
convictionB	conundrumBcontributesBcontradictoryBcontentsB
contagiousB	contactedBconsumesBconsiderablyBconjureBcongressmenBconfrontingBconformBconcurBcompsB
componentsB
complacentBcompilationBcomparesBcompanysBcommonsBcommentariesBcomicalBcombosBcolumnBcolumbiaBcoloursBcologneB
colleaguesBcohortsBcoasterB	clubhouseBclinicalBclearerBcleanupBclayB
citycountyBciteBcitadelBcinnamonBcigsBchunksB	chucklingBchoreB	china’sB
chickenpoxBchewB
cheekbonesBcheckoutBchasedBcharmsB	charlotteBchariotBchargerBcharacterizeBchapotraphouseBchabadB	certifiedBcellsBcelebsBcelebritiesBcbbB
cautiouslyBcaucusBcarverBcardiacBcaptionsBcapesBcapabilitiesBcandlesB	campaignsBcamaraderieBcalorieBcabinetBcabinBcabbageBbyteBbuysBbutcherBburstsBbungieBbunchesBbuhBbuffetsBbuckoBbrushesBbroncoBbroaderBbrigadesBbrieflyBbreedsBbreakthroughBbraverBboyoBboxersBboutsBbourgeoisieBbouncedBbooooBboondockBboltsBbodilyBblurBbloopersBbloomerBbloomBblogsBblinkedB	blessingsBblacksB
blackhawksBblackedBblBbkBbitrateB	biologistBbinaryBbewBbeverlyBbeverageBberyBberlinBbellevueB
belittlingBbelittleBbehavingBbegoneBbeetlesBbeautiesBbeatlesBbeatifulBbeardsBbealBbeadsBbb20BbaristaBbardB	barcelonaBbarcaBbarbedBbarbaricBbantzBbankedBbanffBbaloneyBballingBbaitedBbackstopB
backhandedB	backcourtB	bachelorsBazzBayeeBaxisBawaitBaveragesB	avalancheBavailB	autopilotBautocorrectBautistsBaussiesBauntsBauctionsB	attendingB	attendantBasukaB	assigningBassignBassessmentsBassassinationBaspergerBaslB
articulateBarteryB
arrowverseB	arroganceBarrivingBarmpitBarchiveBaquamanBapronBappropriatingB
approachesBappreciativeB
applaudingBapiB	apatheticB
antivaxersB
antisemiteBantiscienceBantinuclearBantinatalistBangelicBanarchocapitalismBanalystB
ambivalentBamazingiBamateurBaltercationBalphabetBalmondsBallwaysB	allergiesBalgebraBalbeitB	airbenderBailmentsBaggressivenessBafloatBaffiliationBadviseB
advertisedB	adversaryB	admissionBadmiralBadjacentBadidasBadhereBadeptBadditionallyBacreBacopiaBachesBaccusingBabtBabathurBabandonB9kB999B8amB88B7amB77B6070B600msB53B52kB511B5050B442B40kB3pmB3dsB3800B365B31mB314B310B3035B2dayB25000B220B2022B1mgB1995B18thB18kB175B1617B15mgB155B13kB11amB111B1015B100000B🦈B🥂B🤯B🤣iB🤔🤔🤔B🙏🏻B😭iB😘😘😘B😖B😅😅😅B🗑B💭🙏B💚B👸🏻B👊B"🎶💃oww💃mah💃layg💃🎶B🍻B🍕B🍀B-❤🏳️‍🌈🏳️‍🌈🏳️‍🌈B⛑B⛏️💎❤️B☝️B▀B•jumpsB	“yes”B“yeahno”B“wouldB“whereB“unnoticed”B“tripolar”B
“this”B“targets”B“startB
“respectB	“owningB“ourB
“musclesB
“meme”B
“love”B“lolB“liberals”B“itB“intellectuallyB“hoovered”B“hisB“heheB“guest”B“enjoyB“druggies”B	“directB“carveB“breathe”B“beerB“bae❤️”B“anythingB
’good’B‘myB‘kiddyB‘inB‘forB
‘dish’B‍♀️‍♂️‍♂️‍Bג׳‬וליאןB	͜͞ʖ▀B̶d̶u̶t̶c̶h̶B«whatBzullilyBzoomiesB
zombielikeB	zinfandelBzimmB	zeitgeistBzebraBzaxisBzangetsuBz28ByuriBystsByowB	you❣️Byou“ByouiByoubutByoireByoirByidishByforByesteB	yesssssssByesssByellowtintedByeeshByeehawByeeeyByeeeeeeB
yeeeaaahhhByearoldByearlyByeahhhByawByasByanksByamaokaByaaassBxpostedBxpostBxoxoxBxkcd37BxirBxansBx62x65x65x66Bx200bBwufBwtfffffBwsjBwshlBwrongfulBwrithingBwriteupB	writeoffsBwrestleB	wrenchingBwrathfulB	wowowowowBwowingB
wounded”BwouldbeBworthpurposeBworshippingBworse”BworsensB
worryinglyB	wooooshedBwooooBwondrousBwomen™BwoefulB	withdrawnB	withdrawlBwisshhhBwishmBwiselyB
winterfellB	window•BwindbreakersBwill”BwilliamsBwigbodysuitBwife’sBwifehisB
widespreadBwhoveBwhoppersBwhomeverBwholesome¯ツ¯B	whitewashBwhitestBwhistleblowingBwhispersBwhishBwhineyBwhimpBwhiffBwhassuBwhahBwhaddayaBwe‘reBwetherBwelpiBwellsourcedBwellnessBwellllllitsB	wellknownBwellforBwelcherBweezerBweekoldBweekdayB	websitesoBwebbBweasleB
weakstupidBwbBwayyyyyyBwayyyyyBwauwBwatercolourB	watchlistB	watchableBwaste”BwaspinkBwasallamBwasabiBwartheB	warrantedBwarpedBwarningadvertBwarmlyB	warhammerBwardenBwarcraftBwant❤BwandBwalnutBwallyBwaitwhatB
waitressesBwaitersB	wahroongaBwagonBwaddlingBwadBwaahhaBwaaaaayBwaaaaaaaaaaaaahBvuriBvskillBvoteyouBvomitsBvomitedB
volumewiseBvolleyBvoitB
vodkatonicBvocabBvnBvitriolBvisitughBviolinB
villaspangBvieB
videogamesBvideocassettesB
victoriousBvettingBvestBvesperiaBvertigoBvertebratesBvernonBverminBverilyBverdugoBverbsBveracruzB	venuzuelaBventilatorsBvenerateBveeredBvaultBvatBvariesBvariedBvarBvansBvampireBvaliumBvalidateBvalentine’sBvajBvageneBv3BuyuBuuuuhButopiaButilizedButhewhitebaronButhegingercowBusthereBuspsBuso17BuskraxxB	usernamesBusdaB	uscottgalBurgedBuqBuprisingBuploadsBuphillBupbeatBunworthyBunwillingnessBuntrashyBunsurprisinglyBunspokenB	unscathedBunquestionableB
unpunishedBunprofitableB
unpreparedBunplugBunorganizedBunnessecaryBunnaturallyBunnamedBunmotivatedBunmemorableBunloadB	unlikableBunkindBuniversallyBuniquelyBunionistB
unintendedBunifiedBunholyBungenderedmixBunfuckinbelievableB
unfilteredB	unfavoredBuneditedBundueB
undressingBundiesB
undertakenBunderstatedBunderratedlyBunderminingB
undergradsBunderestimatingB
undagrunnnBunconvincedBunconventionalBuncontrolledBuncontestedBunconsitionalBuncomfyBunclesBunchangeableBunavoidableBunavailableB
unarchivedB	unanimousBumyesBumeatheadvernacularB	umbilicalBumassachoositeBulurkerturndcommenterBukeypuncherBujellyjellyspaceBuhthisBuhnoBuhmBuglinessBughterribleBughhBugghhhBueightroundsrapidBudgBuclaBuchristiangreyisdracoBucBubpdwBubisoftBuallegedlynerdyBtylerBtwowB	twocolourBtwitsBtweenBtweakedB
twatwaffleBturretBturnipB
turnaroundBturmionBturdyB
turdburgerB
tupperwareB	tunnelingBtumoursBttsBttkBtsscB	tshirt’Btry”BtryhardBtrustsBtruelyB
troomtroomBtrololoBtrollbutBtrioB
trintellixBtrimmedBtriflingBtreyarchBtravelerBtrascendentalBtransphobesBtransitioningBtransgendersBtransformationBtranscendedBtrailingB
traditionsBtoyotaBtouslesBtoulonBtoptiersBtopsoilBtoppledB
toothbrushBtoonsBtoonieBtoolboxBtonesBtokenBtoggleBtofuBtoastingBtoastersBtmorBtldwBtithingBtingeBtimessssBtimeskipBtiltedBtiltBtidyingBtidingsBtiBthumbsB
thumbnailsBthtsBthronesB	threeteamB	threatensB	thrashersBthoughtfulnessBthoughiBthnxBthinskinnedBthinsBthieveryBthielB	they‘reBtheysignBthereiBthereforBtheoughBthemwaitB
themslevesBtheeeBthatsnotB
thatit’sBthatfeltBthanosBthangB
thajarseffBtesticalBterrifyinglyB	tentativeBtensionsBtennBtenantsBtemptedBtemplesBtellnoBtelkomB
televisionBteemoB
techniquesBteasingBteamlemoncreamsB	teamfightBtdgBtdeeBtariffBtammyB	tailoringBtaglinesBtaggingB	tacticianBtabooBtabletsB	tabarnackBt3BsyntaxB
sympathiesBsymbolicallyBswungBswishyBswippinBswfBsweatingBsweatieB
swearengenBswatBswamp»BsvBsuspenseBsusceptibilityBsurvsBsurroundingsBsurgerysurgeriesBsurdB	supremacyBsupremacistBsupporterstheB	superfundB	supercutsBsunderlanderBsuitorsBsuitesBsuitedBsuingB
suggestiveBsufficeBsue’sBsueeBsuccessfailureB
succeedingBsubstantiveBsubcommitteeB	subbreditBstuporBstuntedBstryfeBstrungB
strikeoutsB	stressorsB	stressingB
streamlabsBstrategicallyBstoredBstopgapsB
stochasticBstirBstimulatingBstill”BstillingBsteroidBsternB
steelyeyedBsteamsBsteamingBstarvedBstarlordB
stargazingBstarfordBstaredBstaplesBstandoffishBstancesBstamkosBstallionBstaleBstakingBstakesBstahmsBstaffordBstacksocialBsstaaaaahhhhppppBssssssssssssssshhhhhhhhhhhhBssssshitBsssholeBssrisBssiBsryyourpartyssolameBsriBsrengthBsquishyBsquidgeBsquidBsqueekyBsquealBsproutsB	spreadcanBspotify’sB
spoonbillsBspookyBspooksBspokaneBspillsBspikeyB	spellingsBspeedersBspecsB	specimensBspecializesBspawnsBspateBspasticatedBspasticallyBspankingBspaceflightBsovietsB
soundcloudBsoulreadBsorceryBsoppyBsoooooooooooooooooBsoon™Bsong😂😂BsongwritersBsomethingedBsomanyBsohowB	sodomitssBsocketsB	sociologyB	societiesBsocalledBsoakedBsnuggledBsnugBsnuckBsnriBsnowsB
snowedicedBsnowbankBsnoozeBsneakblendingBsmuglyBsmugglerBsmoooootthhhhhhhBsmokBsmiledBsmegmaBsluttyBslutsBslumpingBslugBslogansBslitB	sliiiiickBslcBslayerBslayB	slatheredB
slanderingBslamsBskyzoneBskywayBskytrainBskypeBskripalBskittlesBskitsBskinnierBskinnedBskillgapBskamBsixthBsittersBsitcomBsinusesBsinusB	sinkholesBsinglingBsingleplayerBsimiliarBsimBsilveredBsilentlyB
silence”BsilBsigningdraftingBsignaledBsidingBsickestBshynessBshuuuuutBshunnedBshugokiB	shudderedBshrimpsngritzBshrillB	shreddingBshredderBshovedB	shortstopB	shortenedBshortageBshoppersB!shooooooooooooooooooooooooooockerBshoddyBshittttB	shittiestB
shittasticBshithead’sBshinsBshinobiBshiiiiitBshhhhhhhBshelfBsheedBshedsBsharerB
shamefullyB	shaihuludBshagsterB
serializedB
sentencingBsensitivityBsellersBselfselectingBselfrespectBselfinsertwishBselfimportantBselfevidentBselfdeprecatinghumorBselfdefeatingBselfcontrolBselfcontradictoryBseizuresBseizedBseedsB	seductionBsecuringBsectsB	sebastianB	season”B	season’BseasomsBsealedB	scritchesBscrewdriversB
screenplayBscratchyB
scratchingBscoutBscornBscootsBscoobyBschtickB
schoolgirlBscholomanceBscholarshipsBschmancyBschlomoBschizophreniaBsceneryBscarletBscapeBscandinavianBsbsBsaviourBsavannaBsaucesBsatisfyBsatansBsaskatchewanBsaquonBsaodBsaoB	sanitizedBsandyB	sandstormB	sandpaperB	salvationBsalivaBsalinasBsalesmanBsalallahualaihiBsaddensBsacrilegiousBsaameBréalizeBrwewantplatesBrvegetarianBrvBrushesBrunwaysBrunnersBruneBrumoredB	rumblingsBruinerBrugsBrugBrudiBrublevBrtworedditorsonecupBrtrueredditBrthanosdidnothingwrongBrtdB
rsynthwaveBrsiBrsarcasmBrrightBrrantBrpunsBrpragerurineBrpoliticalrevolutionBrpersonalfinanceBroyaltyBrowanBroutingBroutBroundedBrottiesBrotateBrotaryBrosslynBrossB
romance…BrollsafememejpegBrollinBroiB
rockfandomBrockedBrobustBrobsB	robovoiceBrobesBroachB
rnoisygifsBrniceBrnewyorkmetsBrmildlyinterestingsBrmildlyinfuriatingB	rmedicineBrlostredditorsBrllyBrlibertarianBrjokesBrivetingBrivaldigBriskingBrippledBrintjBrincestBrigorousBrightwingersBrightlyBriffBrichesBriamatotalpieceofshitBrgcdebatesqtBrgaybrosBrforwardsfromgrandmaB	rforhonorBreworksB
reviewableB	reversingBrevelingBrevaniteanimeBreusingBreusedBreunitesBreuniteB	retrieverBretrieveBretreadsBretractBretoldBretiresBretardednessBresponsiblyBrespectivelyBreservesB
rescindingBresB
requestingBreqBrepsB
repressionBreporecoveryBrepealedB	repaymentBrentsBrenoBrenewingBrenewB	renderingBrenamedBremixesB	reminisceBremindmebotBreluctantlyBreligionunionistBrelayB
relateableBrelantionshipsBreinstalledBreincarnationBregualrBregrettableB	regretfulBregistrationBregionalizedB
refutationBrefundsB	refreshedB
reelectionBreefB
reeducatedBredskinsBredoingBredneckB
rediculousB
reddit’sB	redditingB
recruitingBrecruitB	reconnectB
recomendedB
reciprocalBrecipesBreceiptsB	recappingBrecapBreboundsBrebloodicansBreassuranceBrealisationsBrealignBreadilyBrcringeB
rchildfreeBrcasBrboneappleteaBrbathswithdoorsBrbabyelephantgifsBravensBravenousBratheistBratemesBrastyBrashBraptorB
rappellingBrappB
rantijokesBrangerimBrandoB	rammsteinBramificationsBramdomBralphBraisesBrainingBrainbowsBraidB	ragecomicBraffBradicalizedBradfemBrabbidsBr49ersBqwandorBquothBquipsBqueuesBquarterbackBquarryB
quantitiesB
qualityandBqfBp’sBpysykBpydBpwnedBpveBputzBpurposesBpurifersBpupilBpumpkinsBpullupsBpukeBpuffyBpuertoBpublixBpublicationB
publicallyBpsychedelicsBpsychedBpseudoBproxyBprovokeB	provisionBprovablyBprotagonistsB
prosperityB
prosecutesBpropablyBpronounciationBpromptedBpromotionalBprojectionsBprohormonesBprogressionsB
programmedBproducerBprodigyB
proclaimedBproassadBprisonsBprioritizedBpridefulB
preventingB	preventedBpretendsBpresentableBprescriptionsB
prescribedB	preschoolBpreppingBpreorderB
prejudicedBprefaceB	predetorsB
precociousB
precedenceBprecanaB	preachingB
pragmatismBpragBpractice…BpoutingBpouncesBpouliotBpotcoinB	postmatchBpostmaritalB	possibleyBportersBpornsB	populistsBpopulistBpopperBpoopingBpoohBponziBponyBpolygonB
polygamousBpolyamorousB
politicizeB
polandballBpointlolBpointerBpoepleBpl’sBplushieBplumbusB	plotlinesBpleromaBpleasantriesB
playstylesB	playmakerBplaydespacitoBplayboyBplayboiBplawBplatypusBplantedB	plaintiffBpixelsBpitsBpitchedBpistonBpiracyBpiqueBpinkersBpingsBpilledBpikeBpigeoness”BpicturesqueBpicnicBpickpocketsB
pickpocketBpickeksB
pick6scoopBphone”BphobicBphilosophicalBphiladelphiaBphasesBphBpettingB	pertinentB
persuasionB
persistentBpersianB
perplexingB	perplexedBperpetuallyBpermissibleB	performerBperfectionismB	peregrineB	pepperoniB
peoples’BpeoplesocietyBpeopeBpennsylvaniaB	penalizedB	penalisedBpelvicBpeevesBpeeringBpedosB
pedophilicBpedesBpedalingBpeasizedBpearlsBpeachesBpdiceBpayoffsBpaydayBpawnshopBpatronizingBpatriotBpathosB
pathfinderBpastriesB
passrusherBpart”BparttimeBparticulateBparticipationBparseB	parrotingBparlayBparkaBparityBparishionersBparaphernaliaBparamterB	paralizedBparadedBparB	papaoutaiBpants”BpantheraBpanicsBpandemicBpaloBpallingB
palliativeBpalehosBpaintingsdepictionsB
painkillerBpagingBpagedBowwBown”B	owhhhhhhhBoverwroughtBoverwhemingBoverstimulationB	oversightBoverpaymentB
overlookedB	overheardB
overcomingBouts”B	outskirtsB
outplayingBoutnextBoutlinesBoutlazyBoutlawsBoutlastBoutkastBoutgoingB
outfielderBoutedBoutcastsB	outburstsBoutageBour’BotrBospreysBoryaBorrrrrBorphanBornaciaBormondBorganicBorbitB	opressiveBoponentsB
operationsBoperatesBoooooohBoooohweeBoooochBoohhhhBoniBonethirdlifeBonebutBoncehereBok”BoilerBohyuckBohareB
oftentimesBofnbeingBoffloadBodspBodeonBoddhowBoctorokBochilsBoceaniaBoccuredBobstructBobsidianBobserveB	obscenelyB
obligatoryBobjectivityBobamaBnylanderBnvBnutterB	nukevilleB	nuglettesBntsBnremtB	noworriesBnotoriouslyBnotionsBnothing’sBnothingsB
noteworthyBnoshaveBnorseB
normalisedBnorilskBnoooooBnonunionB
nontaintedBnonsarcasticBnonmonoBnoninflammatoryBnonhumanB
nonhatefulBnonexistantBnonessentialBnomsBnomineesBnohomoBnodBnobsBno1BnmomndadnbrotherBnmentalBnlBnirvanaBninetiesBnineBnimbysBnightmaresseriouslyBnigeriaBniceiBniaBngosB	nextlevelBnews”BnewsforBnewishBneutralwoodsyB5neurosurgeonentomologistradiologistanesthetistdentistBnetflix’sBnetflixsBnesquickBnerfsBnerdierBnephewsBneoplatonismBneonaziBneedntB
needlesslyBnecksBnecessitiesBnbcsnBnazismBnationalistBnatBnascarBnarcissist”BnappingBnapkinsBnapalmBname²BnametagsBnamecallingBmyyyB	mysteriesBmx9Bmw2BmvpsBmuzzBmutesBmusternoBmussyBmuscularBmusclyBmurrayfieldB	murdereraBmunchBmummiedB	multitudeBmultiplyBmtgB
mrsfrizzleB
mozzarellaB	mouthwashB	motivatorBmotherfukersBmotherfukerBmotelBmossBmorphedBmoresoB
moralisingBmootBmoonwolf1799BmonumentBmontyB
monticelloB
monochromeBmonkeyblackhawkBmonkBmoney”BmoneyzB	monetizedBmomemotionsBmolotovB	molestingBmoistureBmoidBmohawkB
moderatingBmobilityBmmrBmmorpgBmladyBmixtureB
mitigationBmitchyBmitBmisunderstandwhyBmistsB
mistakenlyBmisspellB
missionaryB	mislintatBmisleadBmisdiagnosingBmisanthropyBminorsBminneapolisB
minimizingBminimizeBminimapBminersBmimicsBmilquetoastBmigrantrefugeeBmiddecemberBmicroplasticsBmicrobladingBmgtowBmgsBmgoodboyBmethheadB	metaphorsBmetaphorpremiseBmessesBmerryBmeritBmercifulBmercenariesBmeowingB
menzingersBmemorizeBmelodramaticallyB
melodesignB
meiwinstonBmehubbyBmediterraneanBmedecineBmechanicBmeasuredBmean…BmeantheyBmeans🤷‍♀️BmeaniBmeandB	mcfuckingBmazdaB	mayeatherBmaximizeBmaustonBmatthewsB	materialsBmastodonBmastersBmassifB	massawaaaBmassacreB	masochistBmaskedBmasculinity”BmartiniBmartialB	marriagesBmarlasBmaritalBmarinesB
margaritasB	margarineBmarblesBmansplainerBmanningBmankindB
manhandledBmandateB
manageableBmalwareB	malignantBmakingeditingBmajoringBmaintainingBmainingBmailsB	magickingB	magatardsBmaduroBmacwhateverBmacaroniBlwymmdBlwxBlustsBlurkedBluringBluggageBludicrouslyBluckiestBluckedBluciaBlubeBlsdBlsatBlrsBlowriesBlowhangingfruitBlowersBlowcalBlovehateBlovebombingBloungeBlotsaB	looooooooB	loooooongBlondonerBlolaBloiltyBloggingBlofiBlockingBlobstersBlngBlmkBlmboBlmaooooooooooooooooooooBlma0BlittyBlitreBliteraryBliteralyBliteracyBliterBlisaBlinetesBlinedBlindtB
limpdickedB	limelightBliloBlifespanB	lifedrugsBlibs”Blibertarianism’sBlhwBlfBlessenedBleprosyBleopardBleftwingBleechBlecBlebronsexualBlebronBlebanonBleaveremainB
leavemaybeBlearBleanersBlbrBlaygBlayeredBlawlessness—“theBlawfulBlawbringersBlaughter”B	laughablyBlatkesB	latestageBlateriveBlassieBlarpingBlankaBlamboBlafcBladlesBksBkotorBkotkoBkonamiBkombatBkodosBknowtheBknowledgableBknoooowB
knewjerkedBkmtBklokslagBkkbBkinslerBkinksBkindredBkinBkillieBkillemBkids”BkhlBkettlesBkentrosaurusBkembaBkeepersBkebabB	katyushuaBkatilotBkarmahungryBkarensBkarateBkapustaBkangarooBkalorieBkaedeBkabukiBjzkBjustsadBjustinBjustificationsBjustifiableBjupiterBjunkieBjumpshotBjukeBjugglingB
journalingBjoke™B	jokesthisBjokersBjokedBjocastaBjob”BjitterBjestsBjengaBjeebyBjawlineBjaveB	jaundicedBjajajaBjahBjagaB
jackhammerB	jackassesBjabbingBiuBitchyBitbutB
issuewhichBissuedBisprettyBislamophobicBislamicallyBisilBiseriousBisekaiBiscoBirritateB	irrelvantBirelandsBiosBinwardBinvoiceBinvigoratedBinvestorBinvalidatesB	intuitionBintroducingBintolerableBintjaBinterventionsB
interseptsB
interpretsBinternalizedBinterlockedBinterimBintelligencesB	intellectB
integratedBintegerB
insurgencyBinstinctiveBinstanceBinsisntB
insecurelyBinnovateBinmatesBinhouseBinhalesBinforceB	inflictedB
infectiousB
infectionsBinfectB
infatuatedBinfacyBinfactBinexperiencedBinexpensiveBinertBindulgedB	indignantB
indiginousB
indicatingBindicaBindianapolisBindestructibleBindefinitelyBindecisivenessBincredulousBincredibilisBincontrolableBinconsiderateBinconceivableBincompidentBincompatibilityB	inclusionBinclinedBincitingBincisiveB	incessantB	inceptionB	inbetweenB	impreciseBimposedBimperialismB
impeccableB	immolatesB	immersiveBimmBimagingB
illustrateBillusionBifthenBiffyB	idolizingBidolizeBidoliseBidiotssheepB
ideologiesBideologicalB
identitiesBidentifiableBidekB
idealoguesB	idealistsBidahoBidaBiconsBichigoneroblackBi95BhypertrophyBhyelpBhydroBhveBhushedBhurdlesBhunterBhumpingBhumpB
humorouslyB	humongousBhumanoidBhugrBhudsonBhudBhubsBhrtBhoverBhouse”B
housewivesB
housepartyBhourlyBhoundBhotdogsBhorryBhorribeBhorrayBhornetBhorfordBhordesBhopedforBhoopinBhoopBhoomanBhookingB	honouringBhonkiesB	honeymoonB	homicidalBhomeboyBhomB
holidaysmyBhohhhhBhoedownBhoaxBhmcBhkBhitboxesBhistoricBhindiBhilaryBhijabB
highpayingBhighmountainBhighlightedBhighlandBhigeBhhhhBhhBhgcBhfB
herselfherBhero’sBheronsBheroicB
hereticalsB
hereit’sBhepB
hentaipoonBhelsinkiBhelpmou’sBheebyBheckshesB	hecki’mBheckaB
hebephiliaBheavyweightB	heartedlyBhealthcarescienceBhealersB
headphonesBhazelnutBhazBhaven’t😔BhavBhaulingBhaulBhaterBhashtagBhasbroBharpBharnessBharlemBhardshipB	harder”BharB
happens”BhappendparentsBhandwrittenBhandsetBhandoutBhandjobBhammersB
hamburgersBhaltBhalalBhalB	hairbeardBhahhhaaaaaaaBhahawhatBhahahhahahahaBhahahahshshajshaBhagsBhagelinBhaevBhaemorrhoidBhadhaveBhacksBhabitualBhabitabilityB	habanerosB	haaaaaateBhaaaaBgypsiesBgusgasmB
gunslingerBgungB	gumsticksBguiltingB	guildfordBguildBguidesBguidedBguiceB	guendouziBguarantyB
gtreatmentBgrungeBgrumpBgroomingBgroeningBgrindrBgriftersB	grievanceBgreysBgreissBgreggsB	greatnessBgravyBgraspsBgrapeBgrantsBgrandpaerntsBgrandmotherlyBgrandmasterBgrandeurBgrammqrBgradedBgraciousB	governingBgotsBgosplesBgooooBgoonBgooglesBgoofB	goodoperaBgoodestBgoodcanBgoodboyBgooBgonnBgoldplatBgoddddddddddddddBgnomeB
gluttonousB
glutenfreiBglowingB
glorifyingBglobetrotterBglobeB
globalistsBgloamingBglitterBglimmerB	gleefullyBgleefulBglastonburyBglareBgiveawayBgirlsforeverBgirlfriend’sBgimmickyBgigaBghoulsBghoulBghettoizationB
gettysburgBgetafeBgermsBgermansB	geographyBgentlemenweB	genocidalBgenesisB
generstionBgenerationalBgeneralizedBgeewelpBgeeksBgawdddBgatheredBgatesBgatekeppingBgatedBgastroparisisBgaspedBgaspBgaseBgarrisonBgardaBgankBgangstasBganglyBgame’sBgalleonsBgain”BgahBgagaBgadBgaaaahhhhhhhBgaaBfuzzBfuuuuuuuuuuuckingBfuturamaBfusionBfuryBfurthestB	furiouslyBfuqingB
funnyawfulBfulltimeBfullnessB	fulfilledBfukingBfuckyoumoneyBfuckmemoneyBfuckedupBfuckboyBfuckboisBfruitionB	frontpageB
frontlinesBfrom”BfromtheBfriganBfrierB	friendstvBfriendshipsBfriendromanticBfriendoBfresnoB	frenchingBfree”BfreefreebradshawBfranticallyBfranBfptaBfpBfourigBfountainB	fortitudeB	forthwithBforsureBforheadBforgivesBforgetsB	forgetfulBforeshadowedBforeseenBfoodrelatedBfondnessBfoldsBflunkBfloydBflowingBfloraBfloorsBfloodedBflirtationshipsBflipfloppingBflgBfleurB
fleshlightBfleshedBfleshconsumingB
fleshbloodBflayedBflawedBflattenBflanksBflamingoB	flamebaitBflakedBflailingBflagrantBfirstlyBfirearmB
fingernailBfinelyBfinBfillingsBfiletsBfighttoBfifoB
fiddler’BfgBffaBfetishisticBfessingBferrelBferrariBfencesBfemininity”BfelonsBfellarBfeelingunlessB
feeeemalesBfeedigBfeatBfearuggonessBfavesBfaultyBfatterB
fatalitiesB	fastpacedBfartedBfarleftBfaresBfanspeakBfanficsBfamilysBfamilymembersBfallibleBfakerB	fairfightB	fairfieldBfactoBfacetuneBfabricationsBf4B
eyeopeningBextravagantBextraterrestrialBextolB
extinctionBextinctB	exspousesBexsplainBexsistsB	exquisiteBexpungedBexpressionsBexponentiallyBexploitsBexploitationBexplodesBexperimentsBexpensesB	expandingBexmosBexistentialistBexistedoriginatedBexileBexhaustBexfriendBexfianceB
exercisingBexcusingBexcludedBexcitedstressedBexcessivelyBexcerptBevoBevisceratedBevidenceproofBeverywayBevensBevangelicalismBeusBeurosBethnonationalismB	ethicallyBestimateB	establishBessenceBescapedBesBeruptedBerrrrrrrrrrB	erroneousBerrantBeresB	equalizerBepoxyBep1BenvironmentalBentropyBentreBensuresBensueBenragedBenmeyBenlightenedBenjoyhisB	engineersBendgamesB
endedstillB	enchantedBenactedBenableBemulatorB	empiricalBemiratesBemetophobicBemergingBembarrasingBembalmedBemasculatingB
elseworldsB
eloquentlyB	elopementB	elizabethBelitismBeliminationsBeligibilitypaymentsB	elephantsBelectronicsBelectrocutedBelderyB
ejaculatedB
eighteenthB
eichenwaldBeffectedBeeewBeeeeeBedwardBeditorializedB	edinburghBedgingBecstasyBeasyjustBearringBearliestBeagleB	eaaaaarthB	dysgenicsBdyedB	dwindlingBdwellerBdweebBdwarvesBdwarfismBduperBdunnoitBdundeeBdumb”B
dumbledoorBdumbassiBdullsBduggarsBdugBduffBdude’sBdudeeeeBds9Bds2BdrunkennessB	drunkenlyBdruerBdrowBdroughtBdrooledBdrmBdrinkgifB	drinkableBdrenchedBdredgeBdreamyB	dreams”BdramBdoxxedBdown😂B	downwotedBdownwardBdownvotes”BdoughBdouchebustersB
doubtfulheBdoublesBdotsBdortmundBdorBdopeyBdoorwaysBdoorstepBdoofusBdoodBdoobieBdonBdollyBdoinkBdoggamBdnepropetrovskBdm’sBdmcaBdmc3Bdlc2BdividingB	diversionBdivaBditchedB
distancingB	dissonantBdissociationB	disrupterBdisregardedBdisputatiousB	disprovenB
disposableB
dispensersBdisownedBdisorientedBdisorientatingB
disloyaltyB	dislodgedBdislocationB	dislikingB
disjointedBdisinterestedBdisinfectedB
dishwasherBdisgustinglyBdisgracefullyBdiscovertakebackBdiscloseBdisciplinedBdisappearedBdisagreementsB	dinner”BdineinB
diminishedB	dimensionBdimB	diligenceBdiggahBdifferentiateBdifferentcoloredBdiegoBdiededBdickyB	dichotomyBdibsBdiapersBdialingBdiaBdexBdevotionBdevelopmentsBdevaluesBdetectorBdetectB
detatchingBdesvenlafexineB	designingBdesensitizedB
descriptorBderideB
deregulateBderealizationBdepthsBdepressionanxietyBdepressB
deploymentBdeplorablesBdeniesBdemonstrationBdemonstrablyB
demolitionB
demolishesB	democripsBdeludingBdelorsBdelitheBdelBdehydrationBdehumanisesBdegendB	deflatingBdeflateBdefinitivelyB
definentlyBdefiesBdefectBdefecateB	defeatistB	defaultedBdefaceBdeeB	deductionB	decomposeB	decimatedB	decidedlyBdeceivedBdecayingBdecafBdebtsBdebilitatingB
dead™️BdeadeyeBdeaderBdeadassBdckBdbacksB
dayyuuummmBdayxefxbbxbfBdaytonBdaughter’sBdashersBdaresBdankeB
dangnabbitBdangleBdangitBdampBdamnnB	damnationBdammBdamittBdamitBdaggerBdademotionsB	daddickedBcystB	cyperpunkBcylinderheadBcybersquattingB	cyberpunkBcyberbullyingB
cyberbullyBcwlBcvBcustomer’sBcustodyBcumslutB
cummonistsBcultistsBctjB
cryptonaziBcryptofacistsBcrypticBcryinB	crucifiedBcrownsB	crossfireB	crossbowsBcrookBcroniesB
croissantsB
criticisesBcriticBcrippledBcriminyBcriB	creepybutBcreepilyBcreditedBcrashesBcrankyBcraigslistwouldBcraggyBcradleB
crabbycrabBcr50BcrBcprBcplBcpaBcowgirlsBcovfefeB	courtroomBcountiesBcounterpartsBcounterargumentsBcostlyBcosmeticB	corruptedBcorgiB
copypastasBcopilotBcoordinatorsBcoordinationB
cooptationBcoopsBcooeeeB
convertingBcontroversyBcontrollableB
contractorBcontourB
contentionB	constrictBconsistentwinnerBconsecutiveB	conquerorBcongresswomanBcongressionalB
congregateB	conflatedBconfinedBconfidentlyB
conductorsB	conductedB	condemnedB	concludedBconcludeBconceiveBconcededB	comradelyB
compulsiveB
complexionB	completlyBcompletionistB
compensateBcompatibilityBcompareableBcommunicationsBcommunicatedBcommitedBcomedycemeteryB	columbineB
colorblindBcolludesBcolludedBcollegeundergradBcolaBcockyBcocktailBcockroachesBcockatooB	cockatielBcobBcoastersBcncBclutterfuckeddBclusterBclubbingBclowneyBclobberBcllelinBcliffsBclgBclericB	clenchingBclenchBclemBclausesB	classlessBclassistBclaspBclarksvilleBclangingBclackmannanshireBckB	cjjjjjjjjBcizikasBcivilityBcircumventedBcircumcisionBcicadaBchutzpahBchunkyBchudsBchubBchrtBchromieBchristendomBchorusesBchordBchoosyBchoosingbeggersBcholesterolBchiropractorsBchips”BchipperBchippedB
chipotleeeBchinasB
chillibeesBchilledBcherrrryyyyyBcherokeeB	cherishedBchequeBchemoBcheez”Bcheesiest”B	checkmateB	checklistBcheapskatesBcheaplyBchaserBcharlieBchargingBcharacterizationBchapoB	channeledB	change❤BchaaaandlerBcgBcfaBcentralizationBcentBcelebratoryBceaseBcdlBccallingBcavedBcautionBcauliflowerB
catfuckersB
catfishingBcatersBcatastrophizeBcatastrophicBcatastropheBcasanovaBcartridgesandB	cartridgeBcartiBcarpfishingBcarneBcarmanBcaricaturesBcare”BcaremustBcareersBcarcassB	capturingB
captivatedB
capitulateB
capitaliseBcapfriendlyBcanvasBcanuckBcanopyBcannonballsBcannibalisticB	canistersBcancersBcancelsBcamryBcampingBcalloutB
calculatorB
calculatesBcakethBcakeorBcabezaBcabBbyfarB	byesexualBbwoodBbuybackBbutthurtBbutterfingersB
butfinallyB
butcheringB	butcheredB	bursariesBburroBburpsBburnoutBburdensBbunniesBbullishBbugginBbuggerBbuffettBbudgingBbudgieBbudgetsB	budgetingBbucklesBbuckleBbuckarooBbubbyBbrœtherB	brûléesBbrussels”BbrusselsB	brunettesBbruisesBbruiseBbruinsBbro”BbroskiB
broomstickBbroadwayBbroadcasterBbrisbaneB
brillianceB	brietbartBbrickingBbriBbrexiterB	breweriesBbreezyB
breeeeeeakB
breakcrackB	breadtubeBbrazenlyBbravestB
brainstormB	brainlessBbracketBboyfieBboyeBbox”BbovadaB
bountygateBboundsBboulangeBbottledBbothallBbornsBborderlandsB
boozedrugsB
bootlickerBboothsBboobiesBbongB	bolliwoodBboiseBboilerBboiiiBboggedBbofoBboeBbodypillowsBbodyiBbobsBblundersBbloominBbloodpointsB
bloodborneBblockerBblendedBbleepingB
bleepbloopBblancBblackmailersB
blackheadsB
blackboardBbjjB
bitternessB	bitteringB	bitchhhhhB	biographyBbingemasBbillyBbiiiigBbgsBbffsBbffBbf1wowB	better”B	betteringB
betterbackBbesttheB
bestfriendBbestedBberriesBbenderBbeltsBbelpedioBbellybuttonBbelieveryeahBbeleiveBbejudgyB	beheadingBbegrudginglyBbeelineBbecuaseBbeautiful”BbeautifulthoughBbeaningB	beachparkBbdnBbc’sBbchbsvBbbutBbaylessBbattlefrontBbathingB
bastardingBbashedBbasejumpBbaronsBbaronB	barometerBbardsBbarbersBbarbequeB	barbecuesBbarbecueBbaptistB	baptismalBbaptismBbankrupt”BbangorBbangofBbanalB	baltimoreBbalonBballinBbaleBbaitersBbaitandswitchBbahhhhhBbackyardBbackpageBbackamdBbabybelBb8BazpiBawwwwwBawkwardgoofyBawhB	awfulnessBawfullBawakesBavidBaversBaveragedBaveBavalonBavailabilityB
automationB	automatedBautistB	auschwitzBattrocitiesBattributingB	attributeBattireBattestBattainabilityBattachBatlesstBatlantisB	atlanticaBastuteBastoundBastBassedBassaultingabusingB
assaultingBaspiringBaspenBasoiafBaskersBas3BarmedforcesupdateBarmchairBaristotelianismB	argoniansB	arereallyBarcsysBar10BappropriationsB	appointedBappealsB
apologizesB
apologistsBapologisingB
apologeticBapartmanBanxietygeneralBanxiepukingB
antivaxxerBantireligionBantiqueBantipsychiatryBantipetersonBantiperspirantB
antiisraelBantidemoraticBanticonspiracyB
anticipateBantiableismB	annulmentBannualBannnndBanimesB
animal’sBangolaBangels❤️B
anestheticBanecdataBandpeterBanchorBananarchistBamwayBamoungBammoniaB
amendmentsBambientBambianceBamazing1BamaturBalucardBaltoBalteredBalrdyBalphasBalpacaBalludedBalllllllwaysB
allegationBalledgeB	allaroundBalgeriaBalbuquerqueBalbinoBalbanyBalbabBalarmingB
alamogordoBalaBak’sBairingB
ahemrepostBagonizedBaghastB	aggrievedBaffectionateBaffBafabBaerosvisionBaeBadvisoryBadverbB
adventuresBadminsBadminB	adductionB
addictionsBadapterBadaptBacussedBactullyBacronymBacquiredB
acquaintedBacquaintanceB	acomin’B
achievableBaccudentallyBaccessoriesBacceleratingB	absurdityBabstractionBabstainBabsBaboveaverageBaboutsBabolethBabc’sBaarpBaaahBaaaaahBaaaaaaaaaaaaaahhhB99ofB9799B970B966B95thB946B92B90searlyB89thB8991B86B84B8240B80kmonthB800€B8000B7x04B7kB711B6ftB66B640x480B60dayB6000€B60000B5’10B5v3B538B5000thB4manB4hrsB4872B413B40manB404B4034B4020B35mmB350mB350kB3200ishB316B30ishB308B2xB2docB293B2627B24thB2327B224B210B
20whateverB209B2019’sB2018wtfB2016insteadB20082010B2007B1withoutB1992B1970B185B16yrsB1520B14yoB140000B1234B1142B1110B100tB048B02B01122B008B005B🤣🤣B🤠B🚫🚫🚫🚫B🙄😏🤷‍♀️B😰B😧B[😂😂👸👸👏🙌👏🙌❤❤❤💥💥👑👑💎💎💯💯♀️♀️♀️B😂👏🏼B💰B💯💯💯💯💯B💄💋B💄B👩👩👩B🏳️‍🌈B🍩B🌟✨B!加油！在中国你做什么？B	❤❤☺B♾B♪…B♪B☹☹B≈40B€500B€200B“…don’tB“yeahB“wtfB“womxn”B“willfullyB“whine”B“wantsB“voters”B“uhhhhhhh”B“thedickpill”B“testB“teacher”B	“stupidB“spitebasedB“slutty”B	“sit”B“sighB	“shieldB“sheB“screamingB“religion”B“realB“proof”B
“poop”B“overreacting”B“oneB
“npcs”B
“news”B“netflix”B
“need”B“muhB“manB	“listenB“likeB“kicksB
“jinkiesB	“hurrrrB“hotB“heB“hairB“hailB“gloriousB“glassB
“genghisB“friend”B“forB“figuringB“fckB“expensiveB“expectingB“eatsB“dudeB“discuss”B“cucks”B“childrenB
“can’tB“camB“callB“boring”B“blueB“beB“bahahaha”B“atB“amB“acktchyuallyB“6millionB‘tisB‘theB	‘schoolB‘orwellian’B‘needsB‘footballB‘everywhereB‘cuseB	‘big’B‘badB‘06B‘03B‘BzoeBzitsBzipperBziplocBziggyBzerosBzebrasBzealotsBzealotBzanyBzackBy’sByóuBypgByou‘reByousheByokelsByimbyByikersByiffByieldedByieldByheaaaByehaByeastByeahitsByeahhhhBya”ByayeeeeeB
yawnworthyByaaassssByaaasByaaaayBxlB
xenophobicB
xenophobesBxcomBxboneBx5BwyaBwwooooooooooooBww3BwutsBwudBwtf😟☹🤣Bwtf3BwrrbqbBwrongsBwrongestB
writing”BwriteallBwristsBwrinklyBwraithBwqsBwowzayouBwowuBwoulndtBwouldstBwotB
worshipingB	worrysomeB	worrynextBwormsB	worldlandBworkercouldBwordphotoshopBwordbendingBwooshingBwooshedBwoopsBwoooooweeeeBwooooooooooBwoolB
woodsticksBwonderworldBwolfdogBwoesBwoeBwoBwnwBwithholdingBwishhhhBwisestBwiserBwipeoffsBwipedB
winterrainB
winner’sB	windchillBwiltedB
wildrosersBwildlingBwildcardBwikichristianityBwigsBwiggsBwierdBwhy’dB	whyyyyyyyBwhoooooBwhiteysBwhiteyBwhitB	whistlingBwhinnyBwhingingBwhiffedBwhhooooooaaaaaB	wherewhatBwherebyBwhammyBwhaaatBwetlandBweshubekauanB	wentworthBwellresearchedBwellmeaningBwellinformedBwelcomedBweirdedBweimarBweiB	wehrmachtBwehoeBweeweeBweenieBweekthisBweedsBwedlockBwebsBweaveBwearyBwdBwcBwazeBwaxedBwattBwatpBwaterproofresistentBwas”BwasterecyclingBwasteoftimeBwashroomBwashesBwarmupBwarfareBwanB	wallowingB
wallflowerBwalkwayBwalkinBwaiverBwaiveBwabbiesBwaaahB	waaaaaaayBwaaaaaaallllllBvyingBvuBvt1BvrodBvpdBvoltronB
volkswagenBvolitionBvolatileBvoilaBvocallyBvividlyBvivBvisiblyBvirginityeverBviolntBviolifeBvilchisBvikingstheyBvigilBvigB	victoriasB
victimizedBvicesBvh1BvesselBveryunsettlingBversesBverrrrryBverisimilitudeB
verifiableBverdadBverboseBventilationBvengefulB
venezuelanB
vegrevilleBvegBvbucksBvaxenesB	vasectomyBvapidBvapenorthcaBvapeBvanillayeahBvanguardBvamosBvalorusBvacuumBvacayBv012BuuuuughButilizeButahcoloradoBusyouB
usynthpraxB
userbilityBusdBusableBus3B
up🤔🤔BuptightB
upstandingBuppercutBuppedB	uploadingB	upholdingB
upbringingBuotomyreBunwreckBunwarrantedBuntwistBunspecifiedBunsoundBunsocializedB	unsettledB	unsecuredBunrulyBunrecognizableBunrankedB
unpossibleBunparalleledBunobservantBunnervedBunluckinessBunlegitB
unladylikeBunjustifiableBunisexBunironcallyBunipolarBuninterestedB
uninspiredB
unindictedB
unilingualB
unilateralBuniformsBunicornpostingBunhearBunfuckerizedBunfortnatwlyBunfalsifiableBundifferentiatedBunderwhelmedB	undersideBunderreportB	undermineB
underlinedB	undercutsBundemocraticBuncutBuncontroversialBunconditionallyB	unclearlyBunbrainwashBunbelieveableBunbeliavableBunbeleivableB
unattendedB
unassignedB
unaffectedBunabashedlyBumphBumojomartiniBummmmmB	ultrahardB
ultimatumsB
ultimativeBultiBulteriorBulemaBulebronlover23BuhyunlBuhhhhhitBuhhhhhhBuhhclassBuhaBuglierBuffdaBudderlyBucfsBucareeningdirigibugBubookluvr83BubackwoodmenaceBuback2worksoonB
uadamsmithBuaBu955bspBu897w346354365fdddfsB	u1r0nymanBtytBtyreBtwofaceB	turquoiseBturninBtunnelsBtunesBtulsiBtugsBtuckingBtuckerBttheyBttglBtsasBtryoutBtryhuluB
trumptrollBtruestB	truckstopBtroyBtrollxBtrmzsBtrixieBtristateBtripodBtrick”BtricksBtrendsBtrenB	tremblantB	treehouseB	treasuresBtravisBtraumasBtrash”BtrashierBtransparencyBtranslatingBtransingB
transhumanBtransformationalB	tranlatedB	trademarkBtournyB	tourettesBtotalityBtossingB	tormentedBtoriBtootsBtoomyBtonerBtonedBtomorrow”BtomatoesB	toleratesB	toleramceBtoiletsBtoday”BtobacconistBtngBtmzBtmobileBtmiBtimingsBtime”BtimersBtimeoutBtimeofBtimefreedomBtimebutBtiltlesB	tightenedBtiersBtielessBticketyB	ticketingBticBtiandiB
thunderousBthstB	throwawayB
thrombosisBthriftBthreshB
thrashtownBthrashedB	thoughwhyBthoughtlessBthouBthorBthonkBthogBthiughBthis’sBthisthatBthissssBthishaveBthisbackBthirstBthirdfourthBthinking😂B	thingswowBthingslittleBthingsiB
thingcouldBthetBthermateB	there’dBthere¡B	theressssBthereidBthereandBthenfordBthenastiestnateBthemallBtheistsB
theatricalBtheatreBthayBthat😔B	thatthankB	thanklessBtgeBtextingchapchattingfacetimingB	textbooksBtetrisBterrorsBterriersBtermitesB
terminallyBtermdogBterfcideBteresBtequilaB	templatesBtemperamentalBtempeBtellyBtelevangelistB	teletubbyBteeBteddyowen’sB
technicianBtdmBtbonedBtboneBtaxingBtattooedBtarpBtarnishB	targets10BtappingBtankiesBtankieBtangibleBtamponBtalonsBtakashiBtahBtagsB
tacticallyBtacklingBtacB	tabhiddenBt1dB	system”BsystemicBsyringesBsyncedBsymphatizersBsymfuhnyBsymbolsB	symbolismBsymbolicBswoonBswillBswiftlyBsweryBsweetlyBsuvBsuuuuuuuucksB
sustainingBsusceptibleBsusbscribedBsurvB	surrogateB	surrogacyBsurnamesBsuriveBsurebutBsuppressionB	supposingBsuppleBsuperuniqueB	supernovaBsuperficiallyB	superbikeBsunnahB
sunderlandB
summer’sBsuiderstrandBsuffererBsudburyBsuck”BsuccessfullBsucceedwelpBsubvertBsubscriptionsB
subscriberBsubordinatesBsubmodeB
subbredditBstungBstructuralcivilBstrivesBstrippeddownBstringsBstrideB
stretchingB	strenuousB
streamstheB	streamersBstrawberriesB
strategistBstrangleholdB	strainingBstorysBstormsBstoringB
stopactingBstoopidBstoogesBstjBstingyBstilltheBstiflingBstiffB
stickshiftBstickiedBsthsBsteriodsBstereotypingBstepdadBstellarBsteamdiscordBstealthyB	stealableBsteadfastnessBstbxmilBstaunchnessBstatuteBstatuesBstatsdon’tB
stationaryB	statelessBstarfishB	starcraftBstanleyBstankyBstankBstandoffBstalkinBstale”BstairBstaggeringlyBstackexchangeBstaciesBstabbyBsrdB	squishingBsquirmB	squealingB	squeakingBsqueakedBsquad”BspurtBspudBspriteBspreeBsprayingBspoonB
sponsoringBspongeBspoiltBsplinters”B
splashbackBspitzBspitfireBspiritualityBspidersenseBspiceyBspeltBspeedsBspeechwriterBspeculatingB	specificsBspecialtiesBspecialistespeciallyBspasticBspankBspammersBspaldingB	spacewalkB	spacetimeBsoybeanBsowsBsowellBsourcingB
sourcebookBsoulmateBsotheBsortingBsorcererB,soooooooooooooooooooooooooooooooooooooooooonBson”BsonysBsonofabitchB
somenbjornBsombreroBsollyBsolitudeBsolicitationB
softrebootBsoftestBsodiumpentobarbitalBsodiumBsockpuppetsBsocketBsocioeconomicBsocalBsnowmenBsnowfascistBsnogBsnlBsnitchesBsniffedBsnideBsneaksBsneakierBsndBsnatchedB
snarkinessBsnagglepussB
smileyboyeBsmh”BsmartsBsmallpoxBsmalleatBsmacksBsmackingBslwBsluggingBslrBslownessBslowingBsloveniaBslopBslippingBslingBslimierBsliderBsleevesBslaversBslateBslashesBsladeBskywardBskolBskisBskiparachutingBskintBskinningBskillsetBskilletBskewB
skelebuddyBskateboardingBsizzleBsituationalBsissyBsiriusxmBsiriBsippingBsingersBsinewsBsimultaneityBsimpson’sB	sillinessBsilkBsilencedB
signaturesBsighsBsifuBsiftBsidefreeB
sideeyeingB
sicknessesBsibBsiBshysterB	shufflingBshtickB	shriekingB	showgirlsB	showcasesB	shorthandBshortestBshortcutBshoookB
shoehornedBshockzBshoBshiznittlebamsnipsnapsackB	shittyassB
shittinglyBshittierB
shiteatingBshiptB	shinbonerBshillyBshiftyBshiaBshe”BshelvedBshawtyBshaolinBshantayBshampooBshakeshiversBshaitanBshaftsBshadowbannedBshadahaB	shaaallahBsgtBsextingBsexsBsewBsesameBserousBserendipitousBseraB	separatesBsentryB
sentimentsB	sentienceBsensingBsensibilityBsenorBselloboyBselfreflectionBselfinterestBselfidentifyingB
selfhatingB
selfgodismBselectedBseeigBseededB	sectarianBsectBseattlesBseatingB	seatbeltsBseatbeltB	seasoningBseahawksbroncosB	scuttlingBscudB	scrunchedB	scrubbingBscowlBscoutingBscorersB
scoreboardB	scorchingB	scientismBschottenheimerBschmucksBschmeeBscheisseBscandinaviaBsb2sb2BsaytruthBsaysoBsayohBsayingsBsausagesB
saucebasedBsativaBsatchelsBsatan”B	saskatoonBsarongBsarcasticallyBsarBsaquonsBsapaBsansoresBsandwichingBsandowBsanatizeBsame😂😂😂😂B
sameexceptBsaltiestBsaltedBsalivaryBsalineBsalespersonB	salary”BsalarysBsalariesBsalariedBsalaamBsakuraBsailorBsaigonBsafewayBsadistBsacrificialBsabbathBs7BrzachBryeB
rworldnewsB
rwooooshedBruuunnnBrushersBrupliftingnewsB	runningsoBrunbgbbichidicticbgBrumpBrumblingBrules”BrugratsBrugratBrudestBrudelyBrtexasBrteBrsydneyBrsuddenlygayBrsprainedanklesB
rsadcringeBrreligiousfruitcakeBrredsBrpcmasterraceBrowcketsBroutoftheloopBroundingBrougeBrotflBrosierBrosettaBrootkitsBroomatesBromBroleplayersBrokbuddyretardBroidsBroflmaoB	rochesterBrobinBrobidasBrobbingBroanokeBroB	rnonononoBrneetBrmusicBrmomforaminuteBrmlsBrmasturbationlogBrisquéBriskedBrim14andnothingisdeepBright”BrighteousnessBridiculeBriddlesBricottaBricoBricheB
ricescummyBribBrhubarbBrholdmybeerBrhineBrhegedBrharris2020BrgunnersBrgrandpajoehateBrgameofthronesBrfuckyoukarenBrfalconsBrezB	rewritingB	revolvingB
revocationBreversalBrevampBrevB
retweetingB
retrospectBretroactivelyBretributionBretinasB	retentionBretellBretasBretardationBretaliatingBrestoredBresponsibilitiesBresponsabilityBresolveBreskinBresistedBresetBreserveB	resentingBresemblanceBresellB	resectionBrerollB	requiringB
repudiatedBrepubliB	reptilianB	repressesBreppingB
replicatedBrephraseB
repentalsoBrepairsBrepairedB	repackageBreopenedBrentalsBrentalBrenewedBrenameBremiseBreminiscentB
remainiacsBremainerBreloveBrelishBrelentlesslyB	relegatedBreleasewelpBrelationBrelapsedBrektBreinhartB
reinforcesBrehabilitatingBregulationsbutBregrettablyB	regretingBregretedBregBrefugeesBrefrigeratorBrefreshmentosBrefrainBreformedBrefoldBreflectsB	reflectedBreferredBreeeingB
reeeeeeingBreeeeeeeeeeeeBreeeeeeeBreeeeeeBreeeeaaaaaallyBreeatingBredwingsB
rediscoverBredirectBrediculouslyBrecuperatingBrectumB	recreatedB
reconsiderBrecomendBrecoiledBrecognizableBreclinerBreciprocateBrecieptBrecheckBrebingeBrebatesBrebalancingBreasonaBrealllyBreallifeBreactorBrdpBrdelusionalartistsBrdeadbedroomsBrcanadaBrbsB
rbraincelsBrazzBraz0rBrawsBravingBraveBravagedBraunchyBratshitBrataaeBrasedBrantinatalismBramenBrailedBraidedBraggingBrageyBragesBrageinducingBragBradianceBracketBracistsexistetcB
racializedBrachelBraceiqBr8Br3Br2aliberalsBr12BquèbecBquotaBquizBquittedBquestioningtheBquestionbutBqueasyBquawBquartetBqualityoflifeBqtipBqsBpursuingBpurityB	purebredsBpurebredBpunjabiBpunjabB	punchbowlBpunbadBpumpingB
puljujarviBpuljuBpuffingBpuffBpuckersB
publishingBpublicveteranBpubgBptvBpsyopBpsychosomaticBpsiopsBprovokedBprovisionalB	provincesBproverbsB
protectorsBprostheticsB
prosthesisB	prosecuteBproportionalB
prolongingBprojectiontheyB	programerBproductivityBprodsBprocrastinationB	processusBproabortionBproablyBprivatesBpritoiletriesB	prisonersBprioritizingBpringlesB
primaryingB	primariedBpreventableB
prevailingB	pretttttyBpretoriaBpretenseB	presumingBprestigeiskeysB	preservesB	presenterBpreposterouslyBprepositionBprenupBpremisesBpreludeBprekoopB	prejudiceBpreistB	preflexesBpreferencesBpreeettyBpredevelopmentBpredecessorBprecautionsB
precautionBpranksBpractiseBpprBppgBpotomacBpotholeBpotentBpotatosB
postseasonBpostionBpostesB	postbirthBpossessivenessBpossessionsBposibilitiesBportoBportalsBpornshitB
pormotionsB	porcelainBpopulaceBpopesBpootBpoor”BpoorsBpoorlytrainedBpompousBpoliticsjargonBpoligamyB
policemansBpolarvortexB
polarizingB	podracingBpoBpnpB	pneumoniaBpmoBplymouthBplyBplushB	pluralityBplsssBplowingBplohamasBplezB
pleasuringB	pleasuresBpleaseamuseB	playersnoBplantingBplannersBplanktonBpkayersBpivotBpittyBpitchingB
pitchforksBpitbullsBpinchesBpinballB
pillers”BpierceBpickemBphysiologistBphxB
phrenologyBphilanthropyBphdankBphatBphallicBphalanxBpesoBperverseB
pertainingBpersuadeBperpB	perimeterB	perennialB
percentileB
perceivingBperceiveBpeptoBpenultimateB	penthouseBpenisshapedBpenismonstersB	penaltiesBpeerreviewedBpeeingB	pedotacheBpedantBpearsBpeacekeeperBpcsedB	pcmr4lyfeB
payperviewBpattiBpatterBpasspartBpartysB
parties’BparraBparkedBparetoB
parents’B	pardoningBpardonBpapapaparazziBpandorasB	panameňoBpamphletBpamBpalpBpalletswindowsBpahhhBpageauBpaganophileBpaddingBpackagedBpacersBp7BowchieBovldB	overwriteBovertimeBoversleepingBovershadowingBoverpoweredBoverindulgenceBovergeneralizeB
overdunkedBoverdidBoverconfidentBoverbookingBout”B	outwantedBouttheyB
outrage”B	outplayedB	outmentalBoutjackB	outfittedBoutfieldBoutdoorsBoutdaBoustedBouiBoughtaBouchieBotsBotpsBothetB
ostracizedBosteomyelitisBosrsBosloBoscarB
orthopedicBorovilleBorochiBorneryBorientBorgiesB
organizersB	organismsBorganisationBorganicallyBorderrrrBordealBoratorsBop😅BoptionalBoppressionsBopportunityohB	operativeBopenplanBoopsiesBoopBoooooooyikesBooooohhhhhhB
ooooofffffBooooofffBooohhhhBoohhhBon”B	onelinersBonelinerBoncallBomnidirectionalBomgsoBomgggggBomggggBomggBomgadBomensBomegalulBolympicBolinemanBoligarchsponsoredBolicityrelatedBoldrichguysBoldiesBokogieBokieBokiBokeyBohthisBohkBoffyearBofftopicBofficiatingBofficetheirBoffbyoneBodourBodeskBodellBocularB	octopussyB	occlusiveB
occasionalBoccamsBobviBobtainedB
obligationBoakBnyeeeeeeeeh”BnyawBnutritionistB
nuthuggersB	nullifiedBnuketownBnuhBnprBnowfacesupportB	nowbeforeBnoveauBnotsosubtlyB
noticeablyB	noticableB	nosferatuBnoscopeB
normalizedBnormalcyBnooseBnoooooooonnnBnoonesBnoonB
nonseriousBnonsequiturBnononononoyesB
nonnoahideBnonmetaBnonhpBnonequalitarianBnoncustomersBnomineeBnolookBnolanBnofapBnodsBnodeBnoctisBniñoBnixB
nitpickingBnisaBnipsBnipigonBnineteenB	nightmansBnighBniftyBnickelodeonBnibbasBnggaBnewtonsB
newsworthyB	newcomersBnewbornsB
neutralityBnetherBnestleBnestedBneonB
neofascistBneinsBneilBnegateBnegB	nefariousBneatlyBneaBndasBnbmeBnaviesB
nationwideBnasalBnarudoB
narrativesB	narizonesBnamelyBnakamotoBnairBnagrBnagisaBnachoBnabooB
mythswhichBmysticBmysteriouslyBmyspaceBmuttsBmutilationsBmuthafuckahBmutatedBmutateBmutBmustntBmurderstarveB	murderinoB	multitaskBmultiplyingBmuggingsBmsnbcBmslBms3BmraBmr2BmovieiBmouthheBmotteB	motorheadBmother’s”BmosquesBmosqueBmoscatoBmortonBmortalBmoreiyaBmordorBmorBmopedsBmoonwalkingBmoodsBmonzoBmonumentallyBmontrealBmonstrositiesBmonsantoB
monophanieB
moneypicksBmoneygrabbingBmomoBmolotovsBmolestBmodmailB	models”BmodelersBmodeledBmodelbodybuildelsocialmediaBmockedBmmmmjustBmlpBmk9sBmkBmixupBmiuBmisspeltBmisrepresentedBmisquoteBmisogynistsBmishapBmisconceptionlieBmisconceptionBmirrorsB	mirroringB	minoxidilB
minnesotasBminkeBmingleB	minefieldBmind”BmindreadBmindnumbingB
mindlesslyBmindfulnessBmindbogglingB	mimickingB	milksmithB
milkshakesBmilitiasBmilitaristicBmikolBmightilyB
midwesternB	midseasonBmidpanicBmidolB
midcenturyBmid30sBmicrowavingBmicroaggressionBmickeyBmiaBmeyersBmewingBmethyB
methodicalBmetaphorsaviBmeshBmerlinusBmergerBmeowboysBmen”BmenwomenB
mentalistsBmenowBmemingBmeeeeeeBmeeeBmeddlingBmeatyB	meatballsB
measurableBmearsB
meanhumansB
mealbeforeB
meadowvaleBme1BmcwBmccwordBmccreeBmbaBmbBmayorapeboyBmayocideBmayhapsBmathewsBmateyouB	masterfulBmasonBmascaraBmascBmasalaBmarthBmarsBmarredBmarkingsB
marketthatB
marketableBmarkassBmarinersBman”BmanualBmanipulatesB
manifestedBmanifestBmanequinBmandyBmammyBmamesBmaledominatedBmakebelieveBmajestyB
mainsourceBmagnumsBmagnumBmagnetsBmagiB	magazinesBmagamersBmadmanBmacriBmacosBm109BlynchedBluredBlunch”BlumpsBlumpBlumiBlucidBlowwageBlowsBlowlifeBlowiB
lowballingBlovingsB	louisianaBloudestBlottoBlorainBlopsidedBlootingBloooooveBlooooooooooooooooooooveB	looooooolBlooksmaxxingBlookinfBlolandBloinsB
lockerroomBlocationlanguageBlobotomyB	lobbyistsB
lmfaooooooBlmfaooooBllrB	llanberisBlkeBliving™️BlividB
livelihoodBliuB
lithuanianBlitersB
literatureBlistenerBlispBlinersBlinemenB
linebackerBlineageBline26BlinceBlimeBlikeokBlikentoBliiiiiiiiiii🅱iiiiiiiiiiiitB
lighteningB	lightbulbBligamentBliftsB
lifestylesBlifesiteB
lifesavingBlifelikeBlifelessBlifeiBlickedBlicensesBlibtadB
librariansBlibertariansocialistsB	liberatorBlibdemsBlgbtqBlevelheadedBletstalkB	lethargicB
lessweakerBlesbiansBlendingBlendB
legistaiceB
legislatorBlegislativeBlegionsB
legetimacyBlegalyBlegalsBlegaladviceukBlefttubeBledgeBleashesBlcmsBlbpBlawsuithappyB	lawmakersBlawlB	launchingBlatteBlatexBlassoesBlasikBlashingB	lashes”BlaserswordsBlasdBlaoBlanolinBlalaBlaggyBlackedBlacingBlacedBlabelsB	labellingBl337sBl337BkursenaitelsiaiBkungBkrogerlinkedBkrogerBkrocBkrakenBkqlyBkorrkBkontrolBkomodoBknowledgetechniquesBknivesBknifedBklingonBkleenexBkinderedBkillstreaksBkiefBkidneysB	kidnapperBkidignoranceBkhorneBkhan”BkhanBkhakisBkeystoneBketsBketoBketamineBkenBkdvtBkarenBkap45BkangBkanepiBkadriB	justsorryBjurisdictionsBjunkratBjunglerBjune”BjunctionB	jumpscareBjumpsBjumpersBjukesBjudasBjoyousBjoke’sBjokeespeciallyBjohnnyBjoe’sBjoesBjockeyBjnmilBjitteryBjinxedBjingBjiggyBjewelBjethroB
jeopardizeBjeepersBjeeezBjeeemBjcsBjc01BjaxonsBjanusB
jackasseryBjabbathehutlookingBiwobiBitvB	itthoughtBitchesB	italicizeBitalianamericanBiswasBissurBislayBisitsBironedB	iremindmeBirbeBira19191921BiotaBinvoluntarilyB
invitationBinvinginterrogatingBinvestigativeBinvertedB	inventoryB	inventorsB	invasionsBintruderBintrosB	intrinsicB
intrestingBintingBinterviewedBinterventionismBinterpretedBinterpretationsBinternshipsBinternationalsBinternalisedBinterferingBinterferenceB
interferedB	interfaceBinterestingbutBinterentBinterceptionsBinterbreedingBinterBintakeB
instructedB	instituteB
instigatorBinstabilityBinsofarBinsoB	insincereBinshotBinsertedB
inselaffenBinquisitiveBinquisitionBinquireBinonBinnuendoB	innocentsBinnatelyBinlawsBinitiativesB	inhibitedB
inheritingBinheritanceB
infuriatedB	informingBinfluentialB
influencerB
influencedB
infightingBinexplicablyBinexperienceB	inerrancyB	indulgentBindividuallyB
indirectlyBindictmentsBindefensibleBincrementalBincreceBincreasinglyBinconveniencedBincontinenceBinconsistencyBincompatibleB
incivilityB	incidentsBinbevBin2BimprovesB	impromptuBimprisonB
implicitlyBimpertinentBimpermanenceBimperiusBimperiumBimperialismcolonialismB	impeachedB	impactingB
immigratedBimmagineB	imitatingBimdbBimagine😥BillmanneredBillinformedB$illegitimatenogooddastardlybasteriskBillegitimateBiittssssBidolBidleBidiologyBidfB	idealizedBicedBianlBi7Bi20B
hypothesisB
hypoplasiaBhypnosisBhypesBhyperthyroidismBhyperinteligenceBhydrocortisoneB	hydratingBhybridsBhybridBhwoarangBhvBhuskiesB	husbandryBhurlingBhunnedB	hundredthBhummmBhummingbirdB
humiliatedBhumblyBhumbledBhumans7BhumanistBhuffedBhueyBhuddersfieldBhtBhrt”BhqgsBhpeBhousemoneycarB
householdsBhotdogBhost”BhostessBhospitalityBhospitalisedB
horsewomenBhorsethanksBhorrificallyBhorrendouslyBhormelBhordeB	hooveringBhoofballB	honouraryBhonoringB	honorablyBhonesthoweverBhondurasB	homophoneBhomiesB
homeschoolBhomepageB	homeownerB	homegrownB
homecomingB	homealoneBhollyBholdersBhokeyBhogwartsianBhofer”BhoboBhobbleBhoardedBhmmmingB	hitpiecesBhissyBhippieBhintiBhimikoB	hillbillyBhikineetBhijabsB	highspeedB	highscoreBhighlighterBhighhhhhBhighendBhidesBhiariousBhhowBhexB	hestitantBherreBherrBhermaphroditesBhereticsBherbsBhell’sBhellllllBhelllikeBheftyBheelBheatwaveBheartwrenchinglyBheartstringsBheartpressedBheapsB	healafterBheadquartersB	headlinedBhdrBhazeB	hazardousBhauledBhatlessBhateloveBhaskinsBharpiesB
harmlesslyB	hardhommeBhardhesBhappy”BhappysBhappyinloveB	happen”BhapasBhanzosBhandsawBhandmaidBhandilyBhandheldBhandbookB	hamstrungB
hamsteringBhamptonB
hallelujahBhalffinishedB	halfbloodBhalepBhakedB	hairsprayBhahahahahahahahahahahahahaBhagridBhagBhackingBhackersBhackerBhaaaaayBh2kBgynoBgxBgwnBguy”BguyzBguys”BgushedBguruBgurrBgunpointBgungrabbersB	guitaristBguisBguiltedB
guillotineBguess”BgudgerBguanxiBguageB	guacamoleBgtoBgtkB	grumblingBgrubbyBgrowlyellingscreamingB
groupthinkB	ground”BgroudonsBgropedBgrootBgroomedBgroinBgrizBgripingBgrinderBgrilleBgrievingBgridB	greyhoundBgrenadeBgreetsBgreatsB
greaseballBgravy”BgraspingBgraphBgraniteBgramsBgrailB	graduatesBgpuBgovernmentpoliticsBgovernBgoutBgourdsBgot😂😂BgorgieloyalBgoproBgoosesBgoooooodBgood👌BgoggleBgoawayBgoattaBgoaliesBgmodBgmoBglossingB	glorifiesB	glorifiedBgloomBglandsB
gladiatorsBgladderBgladdensB!givingyoucrapfornotlikingbrowniesBgivinBgitmoBgigglesnortingBgiflikeB	gibraltarB	gibberishBgiambiBgettinB	gesturingBgeonosisB
geographicB	genocidedB
generouslyBgelatinBgeicoBgebBgbaBgazesBgatoB
gateway’B
gatekeeperBgaryBgap’sBgangstaBgammonsBgamethrowingBgamesweBgambitBgallonB	galacticaBgains’BfuzziesBKfuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuckB	fuuuuuuckB
fuuuuuckedBfusB	funnelingBfunkingB	funhammerBfulltangBfullestBfufureBfuddsBfuck’sBfuckloadBfuckfuckBfuckednerfedBfuccB	fruitlessB	fruitcartB	frugalityBfroyoBfrostedBfrootBfriskyB
friends”BfrictionthereBfrickinB	frenemiesBfreerBfreelyread’sB
freelyreadBfreeksBfreedBfranzBfranklinB	frankfurtBfoyerBfournierB	fourmslolB	forwardedB
forseeableBforpostBformdamnBformaldehydeBforensicBfoolishnessBfoieBfoftyBfockBfmtBfluidsBfluctuationsBflowsBflopsBfloofBflipsBflippersBflinchBfleesBfleecethreadBflatmateB
flashpointBflappyBflapBfkBfiteBfishesB	firsthandBfirkinB	firestickBfirefightersBfirefighterBfinnaBfinishesBfiningBfingertipcoversBfingerprintsB
fingerouchBfindtryB
financialsBfilibusteredBfightingbutBfiefBfickleBfianceeBfgmBffmiB
fetishistsBfestusBferryBferrahB	ferdinandBfendB
femshamingBfemmeBfemboiB	feloniousBfeintsBfeeyawnnsayyyyBfeedersBfeckinBfecesBfecalB	featuringBfdaBfdBfc5Bfa’sBfavouritismBfavorsB	favorablyBfatiguedBfathomedBfatassB	fasttrackB	farmvilleBfarmsBfarmingindustriallyBfarmedBfargatB
farfetchedB
fareasternBfapingBfapBfan’BfantasisingB
fanserviceBfanfavoritesBfanfavoriteBfanclubBfanaticsB	falsifiesB	fallaciesBfalB
faithbasedBfairsBfairiesB	facultiesBfackinBfacistB
facecampedBfabricsB	eyeshieldBeyerollB	eyelashesB
extremistsBexteriorB	expresslyB	explorersB
exploitiveBexploitableBexplanationsBexperimentationBexperimentalBexpenses”BexpansiónfixBexpandsBexitsBexhaustsBexertB	exemplaryBexcusedBexcitementexeBexceedBexcedrinBewwwBeverything”B
everybodysBevery1BevenlyB
euthanizesBeuterpeB
eurovisionB
europhilicB	ethiopianBethernetBessexB	espionageBernstBergodanBequivalenceB	equippingBequippedBequalizationBepithetB	epicenterBenvynessBenragingB
engrossingB
engineeredB	enfoldingBenfamilBenduredBendallBenchantressB
encampmentB
emthamusedBemsBemptorBemployBemotiveB	emollientB	emissionsBemigrantBembeddedBembassyBembargoBelyriaBelitistsB
eliminatesBelfenBelevator”BelevatedBelectingBelasticBejectedB	eitherwayBeigherBehhhhBeffingBeeyoreB?eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeBeeaBedgeyBedgedBecourageBecjBearthquakesBearnestBearnerBd”B	dystopianB	dysphoricBdynaBdwindledBdwarfsBduuuudeBdutchyBdurationBdupingBduosBdunnBdumpynerdyuglyBdumbnessBdudeprayBdudBduaBdrovesBdroolingBdriscollBdrilBdregsBdraymondB	drawbacksBdrainsBdragonfruitBdqB	downsizedBdownplayB	downhillsBdowndootBdownbackBdoug’sBdoublingBdormsBdoradoB
don’t”BdontiBdonkeysBdonatedBdonaldtrumpactingB
dominatrixBdomgBdomBdoltBdojBdoiBdogmaBdoeseBdodgyashellBdoctoredB	doctorateBdocketB	doccumentBdobtBdnpsBdjsBdivideBdivergedBditaliteBditaliaBdiswaydBdisturbanceB
distressedBdistinctB	distilledBdissuadeB
dissentingBdisproportionatelyBdisposedBdispleasureB
displacingBdispatchB	disparityBdisparitiesB
disownmentB	disordelyBdismemberingB
disguisingBdisengagementBdisenfranchisementBdisenchantedB
discustingBdiscouragingB	discorollB	discordedBdisconnectsBdiscomfortingBdisarmBdisapprovalBdisappearingB	disablingBdisabilitiesBdirectorgladB
diplomaticBdinnersBdiningBdiniBdineBdimpleBdiminishingBdiluteBdilBdigitBdiggB
differencdBdickishBdiavelBdiatomaceousBdialedBdiagonalB	diabeticsBdhsBdhabiBdfwBdfadBdevaluationBdetourB	dethronedB
detentionsB	detentionBdetachedBdesolateBdesignergraphicBdescentBderryBderankedB	depressesBdepreciationBdeportBdependetBdepayB	denver”BdentBdenominatorBdenimBdenigrationBdenigratingBdeniersBdemocratspfBdemocraticallyBdelusionwhyB
delucionalBdeloitteBdellBdeliriouslyBdeliB	delegatedB
delectableBdejectedBdeicedBdegradedBdeglovedBdefundedBdefunctB
defrostingBdeformedBdeforestingBdeflectsBdeflatedBdefinitelydidnotsettleB	deficientB	defiantlyBdefensivelylolB
defenitelyBdefendsBdefendexplainB
defecationB
defamationB
deescalateBdeepthrotedB	decreasedBdecorumB
decisivelyBdeceptivelyBdecentralizationBdeceiveBdecayBdecadentB	debunkingBdebatersBdebaterB	debatablyB
deathmatchBdear”BdeadbeatB
deactivateBdaypartBdaymanBdawnBdaugtherBdateableBdatasetBdashedB	dashboardBdarudeBdarkseidBdapperBdanmBdangersBdanaBdamperBdammmnBdamceBdadjokesBdaaaaamnB	daaaaaamnBd2BcynicismBcymruBcyclistBcycledBcutshapeBcussingBcussesBcushionBcusaBcurrrrrrrryBcurfewBcurdlingBcuppaBcunningBculturelessBcuisineBcueBcuboBcuBctrBcseaBcryptofascistB	crunchierBcrummyBcrumblesBcruiserBcruellyBcrowningB	cromulentB
crocodilesBcrockB	critiquesB
criticallyBcrippleBcringeworthilyBcrimsoneagl3B	crimelordBcrewsB
creepinessBcreepierB	creationsBcreasesBcreamerBcrayonsBcrayonBcrayBcrawlsBcravenBcratingBcrankedBcraftingBcrackpotBcrackinBcpuBcozinessBcowpokesBcowpokeBcovcathBcourtingBcourtedB
courseworkB
coursethisBcoursesBcoup’dBcountrysideBcountrydenaliBcounterpartBcouncillorsBcougarBcostingBcostcoBcorsiBcorrieB
correlatedBcorpsesBcorpsBcornishB
cornflakesB
corelationB
copyrightsBcopperBcoppedBcopaysBcoordinatedB
coordinateBconvertsBcontroversalBcontributorB	continuumB	continualBconnectivityBconksBcongealBconfidentialBconfederateB
conectionsB	conditonsB
condimentsBcondescensionB
concludingBconciousnessB
conceptionB	conceitedBconcealB	computersBcompulsionsBcomptonBcomprehendingBcompoundingB
compoundedBcompostBcompositionalBcomposerBcomposedB
completleyB	completeyB
complainedBcomplacencyBcompetencewellBcompetedBcompetativeBcompesationsB	compelledB
compatibleBcomparabilityBcompanionforBcompactBcomonBcommunesB	commulismB	committeeBcommitsBcommentsyourBcommaBcomingsBcomersB	comebacksBcombustB	combinefaBcombinationalBcombBcolouredBcolossusBcolossalB	colloquiaB	collidingB
collectiveBcolinBcoincidentallyBcoinbaseBcoffersBcoerciveBcoercionBcoddleBcoconspiratorBcockholsterBcocBcoastedBcoachsBcmBclownedBclogsB	clockworkBclockingB	clippingsBclippersBclinkBclingyBclincherBclinchBclimbedBclichesBclearsBclearcutB	clearanceBclaustrophobicBclassconsciousBclapsBclansBclamBclairBcivilizationsBcivicsBcircuitBcirclingBciafbisBchumpBchudBchucklefuckBchrushB
chroniclesBchopBchooksBchonkersB
chokepointBchogochujangB	chocotacoBchoBchlorineBchisoxBchinookBchinchillasBchimpanzeesB
chimpanzeeBchimpBchillyBchillestBchild”Bchildren”B	childneedB
childbirthBchibiBchetBchefsBcheersimdrunkbotBcheerfulnessB
cheekswaitBcheckerBcheapenBcharismaticBcharasmaticBcharactBchappieBchaoxB	changeorgB
chandelierB	chance”Bchampaign’sBchambersBchafingBchabotBcfmeuB
cerebellarB	celluliteBcdBcctvBcboBcbgBcbasB	cavaliersBcausticBcausalBcatcherBcastrateBcashesBcaseyB
cartridgesB	carryoverBcarpoolB	carpentryBcarouselBcarnieBcaressesB
carelesslyBcardioBcarcrashBcapturesBcapslockB
capriciousB
cappuccinoBcapekinoBcanonizationBcangingBcandoB	canceringBcanaryprojectorgBcampedBcamillaBcallusBcalligraphyBcalculationB
cakeshakesBcadburyBc3poBbythB	bypassingBbypassBbwaghhhghhhBbuyinBbuttcoinBbutlifeBbuthanBbussesBbusesBburyingBburriedBburiesBbureaucracyBburdonBbunkerBbumsBbumpsBbumholeBbumblingBbulwarkBbullstBbullshBbullpenBbulbBbuilding’sBbuggersBbuddy”BbuckeyesBbuchBbubBbtBbseBbruh”BbrowsedBbrowseBbrownleeBbrowBbrothterBbroomsBbrokejoblessBbrittleBbrexshitterB	brexiteerBbrewedB	breathersB	breakoutsB
breakawaysBbreachBbrbB	bravolebsBbravadoBbrandyBbrainedBbraidBbragsBbpc157BboyhoodBbowserwobbleB	bowlerno6BbouquetBboth”BboreignBboostingBboosBboorishBboopingBbooooooooomB	boondocksBboomrevolutionBboomhauerboomhowerBbookshelvesBboogersBbonyBbonfireBbonelessBbondageBbombingsstabbingspeopleB	bolivaresBbmBbluuuuuururrrrrrrghhhhhBblurbB	bluntrudeB	bloombergB	bloodworkBblokesBblockbustersBblobBbloatingB
blisteringBblendingBblendBbledBbleckBbleakestBblazersBblazerBblaséBblastersBblasphemousBblaresBblamBblackwebBblackmailingB	blacklistB
blabberingBbi—BbizarroBbixbyBbittenBbitchassBbisphosphantesBbismolBbiscuitBbisbeeBbirthingBbirdbrainlovingBbiphobiaB
biometricsB	biometricB
bigcousinsBbf’sBbfqoeBbethesdaBbest’BbestofBbeserkerBbernoutBbernBberkeleyBberatedBberateBbeongBbenchesBbelzebubB	belongingBbellsBbellletstalkB	believersBbeliefslackBbejeesusB
begrudgingB
beggarsishB
beforecokeBbefoeBbeetBbeeslyBbeckonsBbecausei’mBbeaverB	beatsaberBbeardedBbearableBbeamerBbeallBbealeBbeaksBbeakBbeaconsBbdsmBbb8B
battlestarBbattleseriouslyBbattleforntBbathurstBbathtubBbathedB	batbulledBbastardizedBbasemanBbaseballfootballbasketballBbarringB
barricadedBbarkingBbarkBbantamBbannnerBbankerBbanishedBbangoB	bangin’BballymunBballsackBballotB	balloonedBballedBbakkieBbaileyBbailBbahamasBbahahahahahahaBbahBbaguetteBbagsyBbaghdadBbadgesBbadgerBbackupsB	backtrackBbacktapBbacklogBbackheelB	backfiresBb2BayrBawfulsB	avulsionsBavgBaveragelookingBavBautotuneBautographedBautodirectorBautobiographyB	autisticsBautimsBauthorillustratorBauchB
attributesBattenboroughB	attackersBattachmentsBatoneBatletiBatlBathleticismwhoopsBatackingBasvabBasueluB	asttonomyB	astronomyB	asterisksBassholemoronB	assfriendB
assemblingBassembleB
assaultersB
aspirationB	ashtabulaBasdaBasariB
arrogantlyBaroseBarooooooondBarmorsBarmiesBarkhamBaristocracyBargonautBarealBarduousBarcsBaramspecialBar15sB
apprestartBappeasementB
apparantlyB
apologisesBaotcBanythingwhatB
anymoreitsBantlersBantizionistB
antivawersBantimigrationBantimenB
antigunnerBantifreeBantifeministBanthropomorphizeBanthraxBantennaBansemBankerBangirisBangelesBandujarBandrogynousBandigotmywatchrewardsBanchorsB	analyticsBanalysedB	analgesicBanaheimBanaB	amusinglyBamuseB	amsterdamBamputationsBamphetamineBamirightBaminBamellBamateursBamassBalumniBalteringBalproBalphabeticalBalmightBall’sBallotBallllllBallitsBalleyBallegedBalicornBalessBalertsBaldsBalchoholBakechiBakBairshipsBairplaneBaimlessB	aicheckedBahyuckBahsBaholesBahhhhtheBaheadsupportBagreeiB	agonisingBagmBaghulasB	aggregateBageingBaf”BafuckingmenBafterargumentBafrinnotBafricanBaflBafkBafiB
aestheticsBaestheticallyBaerialB	advocatedB
advertisesB
advantagesB	advancingB
adulterousBadorbsBadmitiB
admissionsBadministratorsB
adjectivesBadheredBaddisonsBadderallBadcB
adaptationBactuallybutBactionsbehaviorBacrylicBachillesBachievementsB
acheivmentB	acctuallyBaccostedBaccordsB
accordanceB
accompliceB	accessoryB	academicsBabuBabsolutesfuckB	absolutesBabouBabortion”B	aborigineB
aboriginalBablationBabilifyBabbyBabandonmentBaaaaaaaaaaaahB	99percentB950B93B912B8700kB8589B844B83rdB809B799B74B73B72hrsB725B710B71B7080B6figuresB6aB69ingB645B630amB6127B612B610B60ishB60hzB600lbB5’1B5v5B5pB5heB550kB540mgB5050sB500lbB4xB4kdsB4headB4chansB499B435B425B42000B41stB40thB40searly
??	
Const_5Const*
_output_shapes

:??*
dtype0	*??	
value??	B??		??"??	                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '       '      !'      "'      #'      $'      %'      &'      ''      ('      )'      *'      +'      ,'      -'      .'      /'      0'      1'      2'      3'      4'      5'      6'      7'      8'      9'      :'      ;'      <'      ='      >'      ?'      @'      A'      B'      C'      D'      E'      F'      G'      H'      I'      J'      K'      L'      M'      N'      O'      P'      Q'      R'      S'      T'      U'      V'      W'      X'      Y'      Z'      ['      \'      ]'      ^'      _'      `'      a'      b'      c'      d'      e'      f'      g'      h'      i'      j'      k'      l'      m'      n'      o'      p'      q'      r'      s'      t'      u'      v'      w'      x'      y'      z'      {'      |'      }'      ~'      '      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'       (      (      (      (      (      (      (      (      (      	(      
(      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (       (      !(      "(      #(      $(      %(      &(      '(      ((      )(      *(      +(      ,(      -(      .(      /(      0(      1(      2(      3(      4(      5(      6(      7(      8(      9(      :(      ;(      <(      =(      >(      ?(      @(      A(      B(      C(      D(      E(      F(      G(      H(      I(      J(      K(      L(      M(      N(      O(      P(      Q(      R(      S(      T(      U(      V(      W(      X(      Y(      Z(      [(      \(      ](      ^(      _(      `(      a(      b(      c(      d(      e(      f(      g(      h(      i(      j(      k(      l(      m(      n(      o(      p(      q(      r(      s(      t(      u(      v(      w(      x(      y(      z(      {(      |(      }(      ~(      (      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(       )      )      )      )      )      )      )      )      )      	)      
)      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )       )      !)      ")      #)      $)      %)      &)      ')      ()      ))      *)      +)      ,)      -)      .)      /)      0)      1)      2)      3)      4)      5)      6)      7)      8)      9)      :)      ;)      <)      =)      >)      ?)      @)      A)      B)      C)      D)      E)      F)      G)      H)      I)      J)      K)      L)      M)      N)      O)      P)      Q)      R)      S)      T)      U)      V)      W)      X)      Y)      Z)      [)      \)      ])      ^)      _)      `)      a)      b)      c)      d)      e)      f)      g)      h)      i)      j)      k)      l)      m)      n)      o)      p)      q)      r)      s)      t)      u)      v)      w)      x)      y)      z)      {)      |)      })      ~)      )      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)       *      *      *      *      *      *      *      *      *      	*      
*      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *       *      !*      "*      #*      $*      %*      &*      '*      (*      )*      **      +*      ,*      -*      .*      /*      0*      1*      2*      3*      4*      5*      6*      7*      8*      9*      :*      ;*      <*      =*      >*      ?*      @*      A*      B*      C*      D*      E*      F*      G*      H*      I*      J*      K*      L*      M*      N*      O*      P*      Q*      R*      S*      T*      U*      V*      W*      X*      Y*      Z*      [*      \*      ]*      ^*      _*      `*      a*      b*      c*      d*      e*      f*      g*      h*      i*      j*      k*      l*      m*      n*      o*      p*      q*      r*      s*      t*      u*      v*      w*      x*      y*      z*      {*      |*      }*      ~*      *      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*       +      +      +      +      +      +      +      +      +      	+      
+      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +       +      !+      "+      #+      $+      %+      &+      '+      (+      )+      *+      ++      ,+      -+      .+      /+      0+      1+      2+      3+      4+      5+      6+      7+      8+      9+      :+      ;+      <+      =+      >+      ?+      @+      A+      B+      C+      D+      E+      F+      G+      H+      I+      J+      K+      L+      M+      N+      O+      P+      Q+      R+      S+      T+      U+      V+      W+      X+      Y+      Z+      [+      \+      ]+      ^+      _+      `+      a+      b+      c+      d+      e+      f+      g+      h+      i+      j+      k+      l+      m+      n+      o+      p+      q+      r+      s+      t+      u+      v+      w+      x+      y+      z+      {+      |+      }+      ~+      +      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+       ,      ,      ,      ,      ,      ,      ,      ,      ,      	,      
,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,       ,      !,      ",      #,      $,      %,      &,      ',      (,      ),      *,      +,      ,,      -,      .,      /,      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,      :,      ;,      <,      =,      >,      ?,      @,      A,      B,      C,      D,      E,      F,      G,      H,      I,      J,      K,      L,      M,      N,      O,      P,      Q,      R,      S,      T,      U,      V,      W,      X,      Y,      Z,      [,      \,      ],      ^,      _,      `,      a,      b,      c,      d,      e,      f,      g,      h,      i,      j,      k,      l,      m,      n,      o,      p,      q,      r,      s,      t,      u,      v,      w,      x,      y,      z,      {,      |,      },      ~,      ,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,       -      -      -      -      -      -      -      -      -      	-      
-      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -       -      !-      "-      #-      $-      %-      &-      '-      (-      )-      *-      +-      ,-      --      .-      /-      0-      1-      2-      3-      4-      5-      6-      7-      8-      9-      :-      ;-      <-      =-      >-      ?-      @-      A-      B-      C-      D-      E-      F-      G-      H-      I-      J-      K-      L-      M-      N-      O-      P-      Q-      R-      S-      T-      U-      V-      W-      X-      Y-      Z-      [-      \-      ]-      ^-      _-      `-      a-      b-      c-      d-      e-      f-      g-      h-      i-      j-      k-      l-      m-      n-      o-      p-      q-      r-      s-      t-      u-      v-      w-      x-      y-      z-      {-      |-      }-      ~-      -      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-       .      .      .      .      .      .      .      .      .      	.      
.      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .       .      !.      ".      #.      $.      %.      &.      '.      (.      ).      *.      +.      ,.      -.      ..      /.      0.      1.      2.      3.      4.      5.      6.      7.      8.      9.      :.      ;.      <.      =.      >.      ?.      @.      A.      B.      C.      D.      E.      F.      G.      H.      I.      J.      K.      L.      M.      N.      O.      P.      Q.      R.      S.      T.      U.      V.      W.      X.      Y.      Z.      [.      \.      ].      ^.      _.      `.      a.      b.      c.      d.      e.      f.      g.      h.      i.      j.      k.      l.      m.      n.      o.      p.      q.      r.      s.      t.      u.      v.      w.      x.      y.      z.      {.      |.      }.      ~.      .      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.       /      /      /      /      /      /      /      /      /      	/      
/      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /       /      !/      "/      #/      $/      %/      &/      '/      (/      )/      */      +/      ,/      -/      ./      //      0/      1/      2/      3/      4/      5/      6/      7/      8/      9/      :/      ;/      </      =/      >/      ?/      @/      A/      B/      C/      D/      E/      F/      G/      H/      I/      J/      K/      L/      M/      N/      O/      P/      Q/      R/      S/      T/      U/      V/      W/      X/      Y/      Z/      [/      \/      ]/      ^/      _/      `/      a/      b/      c/      d/      e/      f/      g/      h/      i/      j/      k/      l/      m/      n/      o/      p/      q/      r/      s/      t/      u/      v/      w/      x/      y/      z/      {/      |/      }/      ~/      /      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/       0      0      0      0      0      0      0      0      0      	0      
0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0       0      !0      "0      #0      $0      %0      &0      '0      (0      )0      *0      +0      ,0      -0      .0      /0      00      10      20      30      40      50      60      70      80      90      :0      ;0      <0      =0      >0      ?0      @0      A0      B0      C0      D0      E0      F0      G0      H0      I0      J0      K0      L0      M0      N0      O0      P0      Q0      R0      S0      T0      U0      V0      W0      X0      Y0      Z0      [0      \0      ]0      ^0      _0      `0      a0      b0      c0      d0      e0      f0      g0      h0      i0      j0      k0      l0      m0      n0      o0      p0      q0      r0      s0      t0      u0      v0      w0      x0      y0      z0      {0      |0      }0      ~0      0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0       1      1      1      1      1      1      1      1      1      	1      
1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1       1      !1      "1      #1      $1      %1      &1      '1      (1      )1      *1      +1      ,1      -1      .1      /1      01      11      21      31      41      51      61      71      81      91      :1      ;1      <1      =1      >1      ?1      @1      A1      B1      C1      D1      E1      F1      G1      H1      I1      J1      K1      L1      M1      N1      O1      P1      Q1      R1      S1      T1      U1      V1      W1      X1      Y1      Z1      [1      \1      ]1      ^1      _1      `1      a1      b1      c1      d1      e1      f1      g1      h1      i1      j1      k1      l1      m1      n1      o1      p1      q1      r1      s1      t1      u1      v1      w1      x1      y1      z1      {1      |1      }1      ~1      1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1       2      2      2      2      2      2      2      2      2      	2      
2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2       2      !2      "2      #2      $2      %2      &2      '2      (2      )2      *2      +2      ,2      -2      .2      /2      02      12      22      32      42      52      62      72      82      92      :2      ;2      <2      =2      >2      ?2      @2      A2      B2      C2      D2      E2      F2      G2      H2      I2      J2      K2      L2      M2      N2      O2      P2      Q2      R2      S2      T2      U2      V2      W2      X2      Y2      Z2      [2      \2      ]2      ^2      _2      `2      a2      b2      c2      d2      e2      f2      g2      h2      i2      j2      k2      l2      m2      n2      o2      p2      q2      r2      s2      t2      u2      v2      w2      x2      y2      z2      {2      |2      }2      ~2      2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2       3      3      3      3      3      3      3      3      3      	3      
3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3       3      !3      "3      #3      $3      %3      &3      '3      (3      )3      *3      +3      ,3      -3      .3      /3      03      13      23      33      43      53      63      73      83      93      :3      ;3      <3      =3      >3      ?3      @3      A3      B3      C3      D3      E3      F3      G3      H3      I3      J3      K3      L3      M3      N3      O3      P3      Q3      R3      S3      T3      U3      V3      W3      X3      Y3      Z3      [3      \3      ]3      ^3      _3      `3      a3      b3      c3      d3      e3      f3      g3      h3      i3      j3      k3      l3      m3      n3      o3      p3      q3      r3      s3      t3      u3      v3      w3      x3      y3      z3      {3      |3      }3      ~3      3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3       4      4      4      4      4      4      4      4      4      	4      
4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4       4      !4      "4      #4      $4      %4      &4      '4      (4      )4      *4      +4      ,4      -4      .4      /4      04      14      24      34      44      54      64      74      84      94      :4      ;4      <4      =4      >4      ?4      @4      A4      B4      C4      D4      E4      F4      G4      H4      I4      J4      K4      L4      M4      N4      O4      P4      Q4      R4      S4      T4      U4      V4      W4      X4      Y4      Z4      [4      \4      ]4      ^4      _4      `4      a4      b4      c4      d4      e4      f4      g4      h4      i4      j4      k4      l4      m4      n4      o4      p4      q4      r4      s4      t4      u4      v4      w4      x4      y4      z4      {4      |4      }4      ~4      4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4       5      5      5      5      5      5      5      5      5      	5      
5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5       5      !5      "5      #5      $5      %5      &5      '5      (5      )5      *5      +5      ,5      -5      .5      /5      05      15      25      35      45      55      65      75      85      95      :5      ;5      <5      =5      >5      ?5      @5      A5      B5      C5      D5      E5      F5      G5      H5      I5      J5      K5      L5      M5      N5      O5      P5      Q5      R5      S5      T5      U5      V5      W5      X5      Y5      Z5      [5      \5      ]5      ^5      _5      `5      a5      b5      c5      d5      e5      f5      g5      h5      i5      j5      k5      l5      m5      n5      o5      p5      q5      r5      s5      t5      u5      v5      w5      x5      y5      z5      {5      |5      }5      ~5      5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5       6      6      6      6      6      6      6      6      6      	6      
6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6       6      !6      "6      #6      $6      %6      &6      '6      (6      )6      *6      +6      ,6      -6      .6      /6      06      16      26      36      46      56      66      76      86      96      :6      ;6      <6      =6      >6      ?6      @6      A6      B6      C6      D6      E6      F6      G6      H6      I6      J6      K6      L6      M6      N6      O6      P6      Q6      R6      S6      T6      U6      V6      W6      X6      Y6      Z6      [6      \6      ]6      ^6      _6      `6      a6      b6      c6      d6      e6      f6      g6      h6      i6      j6      k6      l6      m6      n6      o6      p6      q6      r6      s6      t6      u6      v6      w6      x6      y6      z6      {6      |6      }6      ~6      6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6       7      7      7      7      7      7      7      7      7      	7      
7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7       7      !7      "7      #7      $7      %7      &7      '7      (7      )7      *7      +7      ,7      -7      .7      /7      07      17      27      37      47      57      67      77      87      97      :7      ;7      <7      =7      >7      ?7      @7      A7      B7      C7      D7      E7      F7      G7      H7      I7      J7      K7      L7      M7      N7      O7      P7      Q7      R7      S7      T7      U7      V7      W7      X7      Y7      Z7      [7      \7      ]7      ^7      _7      `7      a7      b7      c7      d7      e7      f7      g7      h7      i7      j7      k7      l7      m7      n7      o7      p7      q7      r7      s7      t7      u7      v7      w7      x7      y7      z7      {7      |7      }7      ~7      7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7       8      8      8      8      8      8      8      8      8      	8      
8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8       8      !8      "8      #8      $8      %8      &8      '8      (8      )8      *8      +8      ,8      -8      .8      /8      08      18      28      38      48      58      68      78      88      98      :8      ;8      <8      =8      >8      ?8      @8      A8      B8      C8      D8      E8      F8      G8      H8      I8      J8      K8      L8      M8      N8      O8      P8      Q8      R8      S8      T8      U8      V8      W8      X8      Y8      Z8      [8      \8      ]8      ^8      _8      `8      a8      b8      c8      d8      e8      f8      g8      h8      i8      j8      k8      l8      m8      n8      o8      p8      q8      r8      s8      t8      u8      v8      w8      x8      y8      z8      {8      |8      }8      ~8      8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8       9      9      9      9      9      9      9      9      9      	9      
9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9       9      !9      "9      #9      $9      %9      &9      '9      (9      )9      *9      +9      ,9      -9      .9      /9      09      19      29      39      49      59      69      79      89      99      :9      ;9      <9      =9      >9      ?9      @9      A9      B9      C9      D9      E9      F9      G9      H9      I9      J9      K9      L9      M9      N9      O9      P9      Q9      R9      S9      T9      U9      V9      W9      X9      Y9      Z9      [9      \9      ]9      ^9      _9      `9      a9      b9      c9      d9      e9      f9      g9      h9      i9      j9      k9      l9      m9      n9      o9      p9      q9      r9      s9      t9      u9      v9      w9      x9      y9      z9      {9      |9      }9      ~9      9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9       :      :      :      :      :      :      :      :      :      	:      
:      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :       :      !:      ":      #:      $:      %:      &:      ':      (:      ):      *:      +:      ,:      -:      .:      /:      0:      1:      2:      3:      4:      5:      6:      7:      8:      9:      ::      ;:      <:      =:      >:      ?:      @:      A:      B:      C:      D:      E:      F:      G:      H:      I:      J:      K:      L:      M:      N:      O:      P:      Q:      R:      S:      T:      U:      V:      W:      X:      Y:      Z:      [:      \:      ]:      ^:      _:      `:      a:      b:      c:      d:      e:      f:      g:      h:      i:      j:      k:      l:      m:      n:      o:      p:      q:      r:      s:      t:      u:      v:      w:      x:      y:      z:      {:      |:      }:      ~:      :      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:       ;      ;      ;      ;      ;      ;      ;      ;      ;      	;      
;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;       ;      !;      ";      #;      $;      %;      &;      ';      (;      );      *;      +;      ,;      -;      .;      /;      0;      1;      2;      3;      4;      5;      6;      7;      8;      9;      :;      ;;      <;      =;      >;      ?;      @;      A;      B;      C;      D;      E;      F;      G;      H;      I;      J;      K;      L;      M;      N;      O;      P;      Q;      R;      S;      T;      U;      V;      W;      X;      Y;      Z;      [;      \;      ];      ^;      _;      `;      a;      b;      c;      d;      e;      f;      g;      h;      i;      j;      k;      l;      m;      n;      o;      p;      q;      r;      s;      t;      u;      v;      w;      x;      y;      z;      {;      |;      };      ~;      ;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;       <      <      <      <      <      <      <      <      <      	<      
<      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <       <      !<      "<      #<      $<      %<      &<      '<      (<      )<      *<      +<      ,<      -<      .<      /<      0<      1<      2<      3<      4<      5<      6<      7<      8<      9<      :<      ;<      <<      =<      ><      ?<      @<      A<      B<      C<      D<      E<      F<      G<      H<      I<      J<      K<      L<      M<      N<      O<      P<      Q<      R<      S<      T<      U<      V<      W<      X<      Y<      Z<      [<      \<      ]<      ^<      _<      `<      a<      b<      c<      d<      e<      f<      g<      h<      i<      j<      k<      l<      m<      n<      o<      p<      q<      r<      s<      t<      u<      v<      w<      x<      y<      z<      {<      |<      }<      ~<      <      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<       =      =      =      =      =      =      =      =      =      	=      
=      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =       =      !=      "=      #=      $=      %=      &=      '=      (=      )=      *=      +=      ,=      -=      .=      /=      0=      1=      2=      3=      4=      5=      6=      7=      8=      9=      :=      ;=      <=      ==      >=      ?=      @=      A=      B=      C=      D=      E=      F=      G=      H=      I=      J=      K=      L=      M=      N=      O=      P=      Q=      R=      S=      T=      U=      V=      W=      X=      Y=      Z=      [=      \=      ]=      ^=      _=      `=      a=      b=      c=      d=      e=      f=      g=      h=      i=      j=      k=      l=      m=      n=      o=      p=      q=      r=      s=      t=      u=      v=      w=      x=      y=      z=      {=      |=      }=      ~=      =      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=       >      >      >      >      >      >      >      >      >      	>      
>      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >       >      !>      ">      #>      $>      %>      &>      '>      (>      )>      *>      +>      ,>      ->      .>      />      0>      1>      2>      3>      4>      5>      6>      7>      8>      9>      :>      ;>      <>      =>      >>      ?>      @>      A>      B>      C>      D>      E>      F>      G>      H>      I>      J>      K>      L>      M>      N>      O>      P>      Q>      R>      S>      T>      U>      V>      W>      X>      Y>      Z>      [>      \>      ]>      ^>      _>      `>      a>      b>      c>      d>      e>      f>      g>      h>      i>      j>      k>      l>      m>      n>      o>      p>      q>      r>      s>      t>      u>      v>      w>      x>      y>      z>      {>      |>      }>      ~>      >      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>       ?      ?      ?      ?      ?      ?      ?      ?      ?      	?      
?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       ?      !?      "?      #?      $?      %?      &?      '?      (?      )?      *?      +?      ,?      -?      .?      /?      0?      1?      2?      3?      4?      5?      6?      7?      8?      9?      :?      ;?      <?      =?      >?      ??      @?      A?      B?      C?      D?      E?      F?      G?      H?      I?      J?      K?      L?      M?      N?      O?      P?      Q?      R?      S?      T?      U?      V?      W?      X?      Y?      Z?      [?      \?      ]?      ^?      _?      `?      a?      b?      c?      d?      e?      f?      g?      h?      i?      j?      k?      l?      m?      n?      o?      p?      q?      r?      s?      t?      u?      v?      w?      x?      y?      z?      {?      |?      }?      ~?      ?      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??       @      @      @      @      @      @      @      @      @      	@      
@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @       @      !@      "@      #@      $@      %@      &@      '@      (@      )@      *@      +@      ,@      -@      .@      /@      0@      1@      2@      3@      4@      5@      6@      7@      8@      9@      :@      ;@      <@      =@      >@      ?@      @@      A@      B@      C@      D@      E@      F@      G@      H@      I@      J@      K@      L@      M@      N@      O@      P@      Q@      R@      S@      T@      U@      V@      W@      X@      Y@      Z@      [@      \@      ]@      ^@      _@      `@      a@      b@      c@      d@      e@      f@      g@      h@      i@      j@      k@      l@      m@      n@      o@      p@      q@      r@      s@      t@      u@      v@      w@      x@      y@      z@      {@      |@      }@      ~@      @      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@       A      A      A      A      A      A      A      A      A      	A      
A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A       A      !A      "A      #A      $A      %A      &A      'A      (A      )A      *A      +A      ,A      -A      .A      /A      0A      1A      2A      3A      4A      5A      6A      7A      8A      9A      :A      ;A      <A      =A      >A      ?A      @A      AA      BA      CA      DA      EA      FA      GA      HA      IA      JA      KA      LA      MA      NA      OA      PA      QA      RA      SA      TA      UA      VA      WA      XA      YA      ZA      [A      \A      ]A      ^A      _A      `A      aA      bA      cA      dA      eA      fA      gA      hA      iA      jA      kA      lA      mA      nA      oA      pA      qA      rA      sA      tA      uA      vA      wA      xA      yA      zA      {A      |A      }A      ~A      A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A       B      B      B      B      B      B      B      B      B      	B      
B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B       B      !B      "B      #B      $B      %B      &B      'B      (B      )B      *B      +B      ,B      -B      .B      /B      0B      1B      2B      3B      4B      5B      6B      7B      8B      9B      :B      ;B      <B      =B      >B      ?B      @B      AB      BB      CB      DB      EB      FB      GB      HB      IB      JB      KB      LB      MB      NB      OB      PB      QB      RB      SB      TB      UB      VB      WB      XB      YB      ZB      [B      \B      ]B      ^B      _B      `B      aB      bB      cB      dB      eB      fB      gB      hB      iB      jB      kB      lB      mB      nB      oB      pB      qB      rB      sB      tB      uB      vB      wB      xB      yB      zB      {B      |B      }B      ~B      B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B       C      C      C      C      C      C      C      C      C      	C      
C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C       C      !C      "C      #C      $C      %C      &C      'C      (C      )C      *C      +C      ,C      -C      .C      /C      0C      1C      2C      3C      4C      5C      6C      7C      8C      9C      :C      ;C      <C      =C      >C      ?C      @C      AC      BC      CC      DC      EC      FC      GC      HC      IC      JC      KC      LC      MC      NC      OC      PC      QC      RC      SC      TC      UC      VC      WC      XC      YC      ZC      [C      \C      ]C      ^C      _C      `C      aC      bC      cC      dC      eC      fC      gC      hC      iC      jC      kC      lC      mC      nC      oC      pC      qC      rC      sC      tC      uC      vC      wC      xC      yC      zC      {C      |C      }C      ~C      C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C       D      D      D      D      D      D      D      D      D      	D      
D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D       D      !D      "D      #D      $D      %D      &D      'D      (D      )D      *D      +D      ,D      -D      .D      /D      0D      1D      2D      3D      4D      5D      6D      7D      8D      9D      :D      ;D      <D      =D      >D      ?D      @D      AD      BD      CD      DD      ED      FD      GD      HD      ID      JD      KD      LD      MD      ND      OD      PD      QD      RD      SD      TD      UD      VD      WD      XD      YD      ZD      [D      \D      ]D      ^D      _D      `D      aD      bD      cD      dD      eD      fD      gD      hD      iD      jD      kD      lD      mD      nD      oD      pD      qD      rD      sD      tD      uD      vD      wD      xD      yD      zD      {D      |D      }D      ~D      D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D       E      E      E      E      E      E      E      E      E      	E      
E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E       E      !E      "E      #E      $E      %E      &E      'E      (E      )E      *E      +E      ,E      -E      .E      /E      0E      1E      2E      3E      4E      5E      6E      7E      8E      9E      :E      ;E      <E      =E      >E      ?E      @E      AE      BE      CE      DE      EE      FE      GE      HE      IE      JE      KE      LE      ME      NE      OE      PE      QE      RE      SE      TE      UE      VE      WE      XE      YE      ZE      [E      \E      ]E      ^E      _E      `E      aE      bE      cE      dE      eE      fE      gE      hE      iE      jE      kE      lE      mE      nE      oE      pE      qE      rE      sE      tE      uE      vE      wE      xE      yE      zE      {E      |E      }E      ~E      E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E       F      F      F      F      F      F      F      F      F      	F      
F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F       F      !F      "F      #F      $F      %F      &F      'F      (F      )F      *F      +F      ,F      -F      .F      /F      0F      1F      2F      3F      4F      5F      6F      7F      8F      9F      :F      ;F      <F      =F      >F      ?F      @F      AF      BF      CF      DF      EF      FF      GF      HF      IF      JF      KF      LF      MF      NF      OF      PF      QF      RF      SF      TF      UF      VF      WF      XF      YF      ZF      [F      \F      ]F      ^F      _F      `F      aF      bF      cF      dF      eF      fF      gF      hF      iF      jF      kF      lF      mF      nF      oF      pF      qF      rF      sF      tF      uF      vF      wF      xF      yF      zF      {F      |F      }F      ~F      F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F       G      G      G      G      G      G      G      G      G      	G      
G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G       G      !G      "G      #G      $G      %G      &G      'G      (G      )G      *G      +G      ,G      -G      .G      /G      0G      1G      2G      3G      4G      5G      6G      7G      8G      9G      :G      ;G      <G      =G      >G      ?G      @G      AG      BG      CG      DG      EG      FG      GG      HG      IG      JG      KG      LG      MG      NG      OG      PG      QG      RG      SG      TG      UG      VG      WG      XG      YG      ZG      [G      \G      ]G      ^G      _G      `G      aG      bG      cG      dG      eG      fG      gG      hG      iG      jG      kG      lG      mG      nG      oG      pG      qG      rG      sG      tG      uG      vG      wG      xG      yG      zG      {G      |G      }G      ~G      G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G       H      H      H      H      H      H      H      H      H      	H      
H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H       H      !H      "H      #H      $H      %H      &H      'H      (H      )H      *H      +H      ,H      -H      .H      /H      0H      1H      2H      3H      4H      5H      6H      7H      8H      9H      :H      ;H      <H      =H      >H      ?H      @H      AH      BH      CH      DH      EH      FH      GH      HH      IH      JH      KH      LH      MH      NH      OH      PH      QH      RH      SH      TH      UH      VH      WH      XH      YH      ZH      [H      \H      ]H      ^H      _H      `H      aH      bH      cH      dH      eH      fH      gH      hH      iH      jH      kH      lH      mH      nH      oH      pH      qH      rH      sH      tH      uH      vH      wH      xH      yH      zH      {H      |H      }H      ~H      H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H       I      I      I      I      I      I      I      I      I      	I      
I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I       I      !I      "I      #I      $I      %I      &I      'I      (I      )I      *I      +I      ,I      -I      .I      /I      0I      1I      2I      3I      4I      5I      6I      7I      8I      9I      :I      ;I      <I      =I      >I      ?I      @I      AI      BI      CI      DI      EI      FI      GI      HI      II      JI      KI      LI      MI      NI      OI      PI      QI      RI      SI      TI      UI      VI      WI      XI      YI      ZI      [I      \I      ]I      ^I      _I      `I      aI      bI      cI      dI      eI      fI      gI      hI      iI      jI      kI      lI      mI      nI      oI      pI      qI      rI      sI      tI      uI      vI      wI      xI      yI      zI      {I      |I      }I      ~I      I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I       J      J      J      J      J      J      J      J      J      	J      
J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J       J      !J      "J      #J      $J      %J      &J      'J      (J      )J      *J      +J      ,J      -J      .J      /J      0J      1J      2J      3J      4J      5J      6J      7J      8J      9J      :J      ;J      <J      =J      >J      ?J      @J      AJ      BJ      CJ      DJ      EJ      FJ      GJ      HJ      IJ      JJ      KJ      LJ      MJ      NJ      OJ      PJ      QJ      RJ      SJ      TJ      UJ      VJ      WJ      XJ      YJ      ZJ      [J      \J      ]J      ^J      _J      `J      aJ      bJ      cJ      dJ      eJ      fJ      gJ      hJ      iJ      jJ      kJ      lJ      mJ      nJ      oJ      pJ      qJ      rJ      sJ      tJ      uJ      vJ      wJ      xJ      yJ      zJ      {J      |J      }J      ~J      J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J       K      K      K      K      K      K      K      K      K      	K      
K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K       K      !K      "K      #K      $K      %K      &K      'K      (K      )K      *K      +K      ,K      -K      .K      /K      0K      1K      2K      3K      4K      5K      6K      7K      8K      9K      :K      ;K      <K      =K      >K      ?K      @K      AK      BK      CK      DK      EK      FK      GK      HK      IK      JK      KK      LK      MK      NK      OK      PK      QK      RK      SK      TK      UK      VK      WK      XK      YK      ZK      [K      \K      ]K      ^K      _K      `K      aK      bK      cK      dK      eK      fK      gK      hK      iK      jK      kK      lK      mK      nK      oK      pK      qK      rK      sK      tK      uK      vK      wK      xK      yK      zK      {K      |K      }K      ~K      K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K       L      L      L      L      L      L      L      L      L      	L      
L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L       L      !L      "L      #L      $L      %L      &L      'L      (L      )L      *L      +L      ,L      -L      .L      /L      0L      1L      2L      3L      4L      5L      6L      7L      8L      9L      :L      ;L      <L      =L      >L      ?L      @L      AL      BL      CL      DL      EL      FL      GL      HL      IL      JL      KL      LL      ML      NL      OL      PL      QL      RL      SL      TL      UL      VL      WL      XL      YL      ZL      [L      \L      ]L      ^L      _L      `L      aL      bL      cL      dL      eL      fL      gL      hL      iL      jL      kL      lL      mL      nL      oL      pL      qL      rL      sL      tL      uL      vL      wL      xL      yL      zL      {L      |L      }L      ~L      L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L       M      M      M      M      M      M      M      M      M      	M      
M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M       M      !M      "M      #M      $M      %M      &M      'M      (M      )M      *M      +M      ,M      -M      .M      /M      0M      1M      2M      3M      4M      5M      6M      7M      8M      9M      :M      ;M      <M      =M      >M      ?M      @M      AM      BM      CM      DM      EM      FM      GM      HM      IM      JM      KM      LM      MM      NM      OM      PM      QM      RM      SM      TM      UM      VM      WM      XM      YM      ZM      [M      \M      ]M      ^M      _M      `M      aM      bM      cM      dM      eM      fM      gM      hM      iM      jM      kM      lM      mM      nM      oM      pM      qM      rM      sM      tM      uM      vM      wM      xM      yM      zM      {M      |M      }M      ~M      M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M       N      N      N      N      N      N      N      N      N      	N      
N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      
?
StatefulPartitionedCall_2StatefulPartitionedCallStatefulPartitionedCall_1Const_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference_<lambda>_11792178
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference_<lambda>_11792184
:
NoOpNoOp^PartitionedCall^StatefulPartitionedCall_2
?
3None_lookup_table_export_values/LookupTableExportV2LookupTableExportV2StatefulPartitionedCall*
Tkeys0*
Tvalues0	**
_class 
loc:@StatefulPartitionedCall*
_output_shapes

::
?P
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?O
value?OB?O B?O
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
`
	keras_api
_lookup_layer
#_self_saveable_object_factories
_adapt_function*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings
#_self_saveable_object_factories*
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
#(_self_saveable_object_factories
 )_jit_compiled_convolution_op*
?
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
#0_self_saveable_object_factories* 
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
#7_self_saveable_object_factories* 
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_random_generator
#?_self_saveable_object_factories* 
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
#H_self_saveable_object_factories*
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator
#P_self_saveable_object_factories* 
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias
#Y_self_saveable_object_factories*
5
1
&2
'3
F4
G5
W6
X7*
5
0
&1
'2
F3
G4
W5
X6*
* 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
_trace_0
`trace_1
atrace_2
btrace_3* 
6
ctrace_0
dtrace_1
etrace_2
ftrace_3* 
* 
?
giter

hbeta_1

ibeta_2
	jdecay
klearning_ratem?&m?'m?Fm?Gm?Wm?Xm?v?&v?'v?Fv?Gv?Wv?Xv?*

lserving_default* 
* 
* 
\
m	keras_api
nlookup_table
otoken_counts
#p_self_saveable_object_factories*
* 

qtrace_0* 

0*

0*
* 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

wtrace_0* 

xtrace_0* 
jd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 

&0
'1*

&0
'1*
* 
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

F0
G1*

F0
G1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

W0
X1*

W0
X1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
C
0
1
2
3
4
5
6
7
	8*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
?
?_create_resource
?_initialize
?_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUEAdam/embedding_2/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv1d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding_2/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv1d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_3StatefulPartitionedCallserving_default_input_3StatefulPartitionedCall_1ConstConst_1Const_2embedding_2/embeddingsconv1d_2/kernelconv1d_2/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_11791690
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename*embedding_2/embeddings/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp3None_lookup_table_export_values/LookupTableExportV25None_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_2/embeddings/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp1Adam/embedding_2/embeddings/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst_6*-
Tin&
$2"		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_11792319
?
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenameembedding_2/embeddingsconv1d_2/kernelconv1d_2/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateStatefulPartitionedCalltotal_1count_1totalcountAdam/embedding_2/embeddings/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/embedding_2/embeddings/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_11792422??
?
^
+__inference_restored_function_body_11792225
identity: ??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference__creator_6167588^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
x
 __inference__initializer_6167197
unknown
	unknown_0
	unknown_1	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6167181G
ConstConst*
_output_shapes
: *
dtype0*
value	B :`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??22
StatefulPartitionedCallStatefulPartitionedCall:"

_output_shapes

:??:"

_output_shapes

:??
?
i
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791994

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_4_layer_call_fn_11792010

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_11791193b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:?????????Ȩ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????Ȩ:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
?
?
__inference_restore_fn_11792166
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?

?
/__inference_sequential_2_layer_call_fn_11791717

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:???!
	unknown_4:??
	unknown_5:	?
	unknown_6:
Ȩd
	unknown_7:d
	unknown_8:d
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?x
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791581
input_3T
Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handleU
Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value	1
-text_vectorization_3_string_lookup_14_equal_y4
0text_vectorization_3_string_lookup_14_selectv2_t	)
embedding_2_11791558:???)
conv1d_2_11791561:?? 
conv1d_2_11791563:	?$
dense_4_11791569:
Ȩd
dense_4_11791571:d"
dense_5_11791575:d
dense_5_11791577:
identity?? conv1d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2a
 text_vectorization_3/StringLowerStringLowerinput_3*'
_output_shapes
:??????????
'text_vectorization_3/StaticRegexReplaceStaticRegexReplace)text_vectorization_3/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_3/SqueezeSqueeze0text_vectorization_3/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_3/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_3/StringSplit/StringSplitV2StringSplitV2%text_vectorization_3/Squeeze:output:0/text_vectorization_3/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_3/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_3/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_3/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_3/StringSplit/strided_sliceStridedSlice8text_vectorization_3/StringSplit/StringSplitV2:indices:0=text_vectorization_3/StringSplit/strided_slice/stack:output:0?text_vectorization_3/StringSplit/strided_slice/stack_1:output:0?text_vectorization_3/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_3/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_3/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_3/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_3/StringSplit/strided_slice_1StridedSlice6text_vectorization_3/StringSplit/StringSplitV2:shape:0?text_vectorization_3/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_3/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_3/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2LookupTableFindV2Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handle7text_vectorization_3/StringSplit/StringSplitV2:values:0Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
+text_vectorization_3/string_lookup_14/EqualEqual7text_vectorization_3/StringSplit/StringSplitV2:values:0-text_vectorization_3_string_lookup_14_equal_y*
T0*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/SelectV2SelectV2/text_vectorization_3/string_lookup_14/Equal:z:00text_vectorization_3_string_lookup_14_selectv2_tLtext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/IdentityIdentity7text_vectorization_3/string_lookup_14/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_3/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_3/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????,      ?
8text_vectorization_3/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_3/RaggedToTensor/Const:output:07text_vectorization_3/string_lookup_14/Identity:output:0:text_vectorization_3/RaggedToTensor/default_value:output:09text_vectorization_3/StringSplit/strided_slice_1:output:07text_vectorization_3/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_3/RaggedToTensor/RaggedTensorToTensor:result:0embedding_2_11791558*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791153?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_2/StatefulPartitionedCall:output:0conv1d_2_11791561conv1d_2_11791563*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791173?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791086?
flatten_2/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_11791186?
dropout_4/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_11791193?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_11791569dense_4_11791571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11791206?
dropout_5/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_11791217?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_5_11791575dense_5_11791577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_11791230w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv1d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCallD^text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2?
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
E__inference_dense_5_layer_call_and_return_conditional_losses_11792099

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
.
__inference__destroyer_6167158
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?B
?
__inference_adapt_step_6166561
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
?

?
&__inference_signature_wrapper_11791690
input_3
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:???!
	unknown_4:??
	unknown_5:	?
	unknown_6:
Ȩd
	unknown_7:d
	unknown_8:d
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_11791074o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
 __inference__initializer_6167174:
6key_value_init1197641_lookuptableimportv2_table_handle2
.key_value_init1197641_lookuptableimportv2_keys4
0key_value_init1197641_lookuptableimportv2_values	
identity??)key_value_init1197641/LookupTableImportV2?
)key_value_init1197641/LookupTableImportV2LookupTableImportV26key_value_init1197641_lookuptableimportv2_table_handle.key_value_init1197641_lookuptableimportv2_keys0key_value_init1197641_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :r
NoOpNoOp*^key_value_init1197641/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??2V
)key_value_init1197641/LookupTableImportV2)key_value_init1197641/LookupTableImportV2:"

_output_shapes

:??:"

_output_shapes

:??
?
?
*__inference_dense_5_layer_call_fn_11792088

inputs
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_11791230o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
/
__inference__destroyer_11792139
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6225066G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
I
__inference__creator_6166489
identity??StatefulPartitionedCall?
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
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6166485`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
ל
?

#__inference__wrapped_model_11791074
input_3a
]sequential_2_text_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handleb
^sequential_2_text_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value	>
:sequential_2_text_vectorization_3_string_lookup_14_equal_yA
=sequential_2_text_vectorization_3_string_lookup_14_selectv2_t	G
2sequential_2_embedding_2_embedding_lookup_11791034:???Y
Asequential_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource:??D
5sequential_2_conv1d_2_biasadd_readvariableop_resource:	?G
3sequential_2_dense_4_matmul_readvariableop_resource:
ȨdB
4sequential_2_dense_4_biasadd_readvariableop_resource:dE
3sequential_2_dense_5_matmul_readvariableop_resource:dB
4sequential_2_dense_5_biasadd_readvariableop_resource:
identity??,sequential_2/conv1d_2/BiasAdd/ReadVariableOp?8sequential_2/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?+sequential_2/dense_4/BiasAdd/ReadVariableOp?*sequential_2/dense_4/MatMul/ReadVariableOp?+sequential_2/dense_5/BiasAdd/ReadVariableOp?*sequential_2/dense_5/MatMul/ReadVariableOp?)sequential_2/embedding_2/embedding_lookup?Psequential_2/text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2n
-sequential_2/text_vectorization_3/StringLowerStringLowerinput_3*'
_output_shapes
:??????????
4sequential_2/text_vectorization_3/StaticRegexReplaceStaticRegexReplace6sequential_2/text_vectorization_3/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
)sequential_2/text_vectorization_3/SqueezeSqueeze=sequential_2/text_vectorization_3/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????t
3sequential_2/text_vectorization_3/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
;sequential_2/text_vectorization_3/StringSplit/StringSplitV2StringSplitV22sequential_2/text_vectorization_3/Squeeze:output:0<sequential_2/text_vectorization_3/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
Asequential_2/text_vectorization_3/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Csequential_2/text_vectorization_3/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Csequential_2/text_vectorization_3/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
;sequential_2/text_vectorization_3/StringSplit/strided_sliceStridedSliceEsequential_2/text_vectorization_3/StringSplit/StringSplitV2:indices:0Jsequential_2/text_vectorization_3/StringSplit/strided_slice/stack:output:0Lsequential_2/text_vectorization_3/StringSplit/strided_slice/stack_1:output:0Lsequential_2/text_vectorization_3/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Csequential_2/text_vectorization_3/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esequential_2/text_vectorization_3/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential_2/text_vectorization_3/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential_2/text_vectorization_3/StringSplit/strided_slice_1StridedSliceCsequential_2/text_vectorization_3/StringSplit/StringSplitV2:shape:0Lsequential_2/text_vectorization_3/StringSplit/strided_slice_1/stack:output:0Nsequential_2/text_vectorization_3/StringSplit/strided_slice_1/stack_1:output:0Nsequential_2/text_vectorization_3/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
dsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastDsequential_2/text_vectorization_3/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
fsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastFsequential_2/text_vectorization_3/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
nsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapehsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
nsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
msequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdwsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0wsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
rsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
psequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatervsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0{sequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
msequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasttsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
psequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
lsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxhsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ysequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
nsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
lsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2usequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0wsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
lsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulqsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0psequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
psequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumjsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0psequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
psequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumjsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0tsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
psequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
qsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounthsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ysequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
ksequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
fsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumxsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0tsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
osequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
ksequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
fsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2xsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0lsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0tsequential_2/text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Psequential_2/text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2LookupTableFindV2]sequential_2_text_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handleDsequential_2/text_vectorization_3/StringSplit/StringSplitV2:values:0^sequential_2_text_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8sequential_2/text_vectorization_3/string_lookup_14/EqualEqualDsequential_2/text_vectorization_3/StringSplit/StringSplitV2:values:0:sequential_2_text_vectorization_3_string_lookup_14_equal_y*
T0*#
_output_shapes
:??????????
;sequential_2/text_vectorization_3/string_lookup_14/SelectV2SelectV2<sequential_2/text_vectorization_3/string_lookup_14/Equal:z:0=sequential_2_text_vectorization_3_string_lookup_14_selectv2_tYsequential_2/text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
;sequential_2/text_vectorization_3/string_lookup_14/IdentityIdentityDsequential_2/text_vectorization_3/string_lookup_14/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
>sequential_2/text_vectorization_3/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
6sequential_2/text_vectorization_3/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????,      ?
Esequential_2/text_vectorization_3/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor?sequential_2/text_vectorization_3/RaggedToTensor/Const:output:0Dsequential_2/text_vectorization_3/string_lookup_14/Identity:output:0Gsequential_2/text_vectorization_3/RaggedToTensor/default_value:output:0Fsequential_2/text_vectorization_3/StringSplit/strided_slice_1:output:0Dsequential_2/text_vectorization_3/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
)sequential_2/embedding_2/embedding_lookupResourceGather2sequential_2_embedding_2_embedding_lookup_11791034Nsequential_2/text_vectorization_3/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*E
_class;
97loc:@sequential_2/embedding_2/embedding_lookup/11791034*-
_output_shapes
:???????????*
dtype0?
2sequential_2/embedding_2/embedding_lookup/IdentityIdentity2sequential_2/embedding_2/embedding_lookup:output:0*
T0*E
_class;
97loc:@sequential_2/embedding_2/embedding_lookup/11791034*-
_output_shapes
:????????????
4sequential_2/embedding_2/embedding_lookup/Identity_1Identity;sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????v
+sequential_2/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_2/conv1d_2/Conv1D/ExpandDims
ExpandDims=sequential_2/embedding_2/embedding_lookup/Identity_1:output:04sequential_2/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
8sequential_2/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0o
-sequential_2/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_2/conv1d_2/Conv1D/ExpandDims_1
ExpandDims@sequential_2/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:06sequential_2/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
sequential_2/conv1d_2/Conv1DConv2D0sequential_2/conv1d_2/Conv1D/ExpandDims:output:02sequential_2/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
?
$sequential_2/conv1d_2/Conv1D/SqueezeSqueeze%sequential_2/conv1d_2/Conv1D:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims

??????????
,sequential_2/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_2/conv1d_2/BiasAddBiasAdd-sequential_2/conv1d_2/Conv1D/Squeeze:output:04sequential_2/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
sequential_2/conv1d_2/ReluRelu&sequential_2/conv1d_2/BiasAdd:output:0*
T0*-
_output_shapes
:???????????m
+sequential_2/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
'sequential_2/max_pooling1d_2/ExpandDims
ExpandDims(sequential_2/conv1d_2/Relu:activations:04sequential_2/max_pooling1d_2/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
$sequential_2/max_pooling1d_2/MaxPoolMaxPool0sequential_2/max_pooling1d_2/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
?
$sequential_2/max_pooling1d_2/SqueezeSqueeze-sequential_2/max_pooling1d_2/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
m
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H?  ?
sequential_2/flatten_2/ReshapeReshape-sequential_2/max_pooling1d_2/Squeeze:output:0%sequential_2/flatten_2/Const:output:0*
T0*)
_output_shapes
:?????????Ȩ?
sequential_2/dropout_4/IdentityIdentity'sequential_2/flatten_2/Reshape:output:0*
T0*)
_output_shapes
:?????????Ȩ?
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
Ȩd*
dtype0?
sequential_2/dense_4/MatMulMatMul(sequential_2/dropout_4/Identity:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dz
sequential_2/dense_4/TanhTanh%sequential_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d|
sequential_2/dropout_5/IdentityIdentitysequential_2/dense_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
sequential_2/dense_5/MatMulMatMul(sequential_2/dropout_5/Identity:output:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_2/dense_5/SoftmaxSoftmax%sequential_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&sequential_2/dense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^sequential_2/conv1d_2/BiasAdd/ReadVariableOp9^sequential_2/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp,^sequential_2/dense_5/BiasAdd/ReadVariableOp+^sequential_2/dense_5/MatMul/ReadVariableOp*^sequential_2/embedding_2/embedding_lookupQ^sequential_2/text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 2\
,sequential_2/conv1d_2/BiasAdd/ReadVariableOp,sequential_2/conv1d_2/BiasAdd/ReadVariableOp2t
8sequential_2/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp8sequential_2/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2Z
+sequential_2/dense_5/BiasAdd/ReadVariableOp+sequential_2/dense_5/BiasAdd/ReadVariableOp2X
*sequential_2/dense_5/MatMul/ReadVariableOp*sequential_2/dense_5/MatMul/ReadVariableOp2V
)sequential_2/embedding_2/embedding_lookup)sequential_2/embedding_2/embedding_lookup2?
Psequential_2/text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2Psequential_2/text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_dense_4_layer_call_fn_11792041

inputs
unknown:
Ȩd
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11791206o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????Ȩ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
?

?
E__inference_dense_4_layer_call_and_return_conditional_losses_11791206

inputs2
matmul_readvariableop_resource:
Ȩd-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ȩd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????Ȩ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
?

f
G__inference_dropout_4_layer_call_and_return_conditional_losses_11791325

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:?????????ȨC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:?????????Ȩ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:?????????Ȩq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:?????????Ȩk
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:?????????Ȩ[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:?????????Ȩ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????Ȩ:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
?

?
/__inference_sequential_2_layer_call_fn_11791744

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:???!
	unknown_4:??
	unknown_5:	?
	unknown_6:
Ȩd
	unknown_7:d
	unknown_8:d
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791455o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_sequential_2_layer_call_fn_11791507
input_3
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:???!
	unknown_4:??
	unknown_5:	?
	unknown_6:
Ȩd
	unknown_7:d
	unknown_8:d
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791455o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

f
G__inference_dropout_4_layer_call_and_return_conditional_losses_11792032

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:?????????ȨC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:?????????Ȩ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:?????????Ȩq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:?????????Ȩk
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:?????????Ȩ[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:?????????Ȩ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????Ȩ:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
?
i
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791086

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_restored_function_body_6225018
unknown
	unknown_0
	unknown_1	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__initializer_6167197^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??22
StatefulPartitionedCallStatefulPartitionedCall:"

_output_shapes

:??:"

_output_shapes

:??
?
?
.__inference_embedding_2_layer_call_fn_11791947

inputs	
unknown:???
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791153u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ڇ
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791835

inputsT
Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handleU
Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value	1
-text_vectorization_3_string_lookup_14_equal_y4
0text_vectorization_3_string_lookup_14_selectv2_t	:
%embedding_2_embedding_lookup_11791795:???L
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:??7
(conv1d_2_biasadd_readvariableop_resource:	?:
&dense_4_matmul_readvariableop_resource:
Ȩd5
'dense_4_biasadd_readvariableop_resource:d8
&dense_5_matmul_readvariableop_resource:d5
'dense_5_biasadd_readvariableop_resource:
identity??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup?Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2`
 text_vectorization_3/StringLowerStringLowerinputs*'
_output_shapes
:??????????
'text_vectorization_3/StaticRegexReplaceStaticRegexReplace)text_vectorization_3/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_3/SqueezeSqueeze0text_vectorization_3/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_3/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_3/StringSplit/StringSplitV2StringSplitV2%text_vectorization_3/Squeeze:output:0/text_vectorization_3/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_3/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_3/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_3/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_3/StringSplit/strided_sliceStridedSlice8text_vectorization_3/StringSplit/StringSplitV2:indices:0=text_vectorization_3/StringSplit/strided_slice/stack:output:0?text_vectorization_3/StringSplit/strided_slice/stack_1:output:0?text_vectorization_3/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_3/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_3/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_3/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_3/StringSplit/strided_slice_1StridedSlice6text_vectorization_3/StringSplit/StringSplitV2:shape:0?text_vectorization_3/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_3/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_3/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2LookupTableFindV2Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handle7text_vectorization_3/StringSplit/StringSplitV2:values:0Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
+text_vectorization_3/string_lookup_14/EqualEqual7text_vectorization_3/StringSplit/StringSplitV2:values:0-text_vectorization_3_string_lookup_14_equal_y*
T0*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/SelectV2SelectV2/text_vectorization_3/string_lookup_14/Equal:z:00text_vectorization_3_string_lookup_14_selectv2_tLtext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/IdentityIdentity7text_vectorization_3/string_lookup_14/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_3/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_3/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????,      ?
8text_vectorization_3/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_3/RaggedToTensor/Const:output:07text_vectorization_3/string_lookup_14/Identity:output:0:text_vectorization_3/RaggedToTensor/default_value:output:09text_vectorization_3/StringSplit/strided_slice_1:output:07text_vectorization_3/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding_2/embedding_lookupResourceGather%embedding_2_embedding_lookup_11791795Atext_vectorization_3/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*8
_class.
,*loc:@embedding_2/embedding_lookup/11791795*-
_output_shapes
:???????????*
dtype0?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_2/embedding_lookup/11791795*-
_output_shapes
:????????????
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_2/Conv1D/ExpandDims
ExpandDims0embedding_2/embedding_lookup/Identity_1:output:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
?
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims

??????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????h
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*-
_output_shapes
:???????????`
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
?
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H?  ?
flatten_2/ReshapeReshape max_pooling1d_2/Squeeze:output:0flatten_2/Const:output:0*
T0*)
_output_shapes
:?????????Ȩn
dropout_4/IdentityIdentityflatten_2/Reshape:output:0*
T0*)
_output_shapes
:?????????Ȩ?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
Ȩd*
dtype0?
dense_4/MatMulMatMuldropout_4/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????db
dropout_5/IdentityIdentitydense_4/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_5/MatMulMatMuldropout_5/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookupD^text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2?
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
X
+__inference_restored_function_body_11792230
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference__creator_6166489^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_11791217

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
.
__inference__destroyer_6167167
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6167162G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
<
__inference__creator_6166481
identity??
hash_table?

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*:
shared_name+)1197642_load_4252035_4252262_load_6166387*
use_node_name_sharing(*
value_dtype0	S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
:
*__inference_restored_function_body_6166793
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__initializer_6166789O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
H
,__inference_flatten_2_layer_call_fn_11791999

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_11791186b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:?????????Ȩ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?E
?
!__inference__traced_save_11792319
file_prefix5
1savev2_embedding_2_embeddings_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop>
:savev2_none_lookup_table_export_values_lookuptableexportv2@
<savev2_none_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_embedding_2_embeddings_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop<
8savev2_adam_embedding_2_embeddings_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_2_embeddings_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop:savev2_none_lookup_table_export_values_lookuptableexportv2<savev2_none_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_2_embeddings_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop8savev2_adam_embedding_2_embeddings_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :???:??:?:
Ȩd:d:d:: : : : : ::: : : : :???:??:?:
Ȩd:d:d::???:??:?:
Ȩd:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:'#
!
_output_shapes
:???:*&
$
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
Ȩd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :'#
!
_output_shapes
:???:*&
$
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
Ȩd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::'#
!
_output_shapes
:???:*&
$
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
Ȩd: 

_output_shapes
:d:$ 

_output_shapes

:d:  

_output_shapes
::!

_output_shapes
: 
?
0
 __inference__initializer_6166789
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
e
,__inference_dropout_5_layer_call_fn_11792062

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
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_11791292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
u
__inference_<lambda>_11792178
unknown
	unknown_0
	unknown_1	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6225018J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??22
StatefulPartitionedCallStatefulPartitionedCall:"

_output_shapes

:??:"

_output_shapes

:??
?
?
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791173

inputsC
+conv1d_expanddims_1_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims

?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????V
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:???????????g
IdentityIdentityRelu:activations:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
E__inference_dense_5_layer_call_and_return_conditional_losses_11791230

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
N
2__inference_max_pooling1d_2_layer_call_fn_11791986

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791086v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791956

inputs	.
embedding_lookup_11791950:???
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_11791950inputs*
Tindices0	*,
_class"
 loc:@embedding_lookup/11791950*-
_output_shapes
:???????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/11791950*-
_output_shapes
:????????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????y
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:???????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_save_fn_11792158
checkpoint_keyD
@none_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??3None_lookup_table_export_values/LookupTableExportV2?
3None_lookup_table_export_values/LookupTableExportV2LookupTableExportV2@none_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: |

Identity_2Identity:None_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ~

Identity_5Identity<None_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:|
NoOpNoOp4^None_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2j
3None_lookup_table_export_values/LookupTableExportV23None_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?{
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791455

inputsT
Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handleU
Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value	1
-text_vectorization_3_string_lookup_14_equal_y4
0text_vectorization_3_string_lookup_14_selectv2_t	)
embedding_2_11791432:???)
conv1d_2_11791435:?? 
conv1d_2_11791437:	?$
dense_4_11791443:
Ȩd
dense_4_11791445:d"
dense_5_11791449:d
dense_5_11791451:
identity?? conv1d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2`
 text_vectorization_3/StringLowerStringLowerinputs*'
_output_shapes
:??????????
'text_vectorization_3/StaticRegexReplaceStaticRegexReplace)text_vectorization_3/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_3/SqueezeSqueeze0text_vectorization_3/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_3/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_3/StringSplit/StringSplitV2StringSplitV2%text_vectorization_3/Squeeze:output:0/text_vectorization_3/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_3/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_3/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_3/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_3/StringSplit/strided_sliceStridedSlice8text_vectorization_3/StringSplit/StringSplitV2:indices:0=text_vectorization_3/StringSplit/strided_slice/stack:output:0?text_vectorization_3/StringSplit/strided_slice/stack_1:output:0?text_vectorization_3/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_3/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_3/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_3/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_3/StringSplit/strided_slice_1StridedSlice6text_vectorization_3/StringSplit/StringSplitV2:shape:0?text_vectorization_3/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_3/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_3/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2LookupTableFindV2Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handle7text_vectorization_3/StringSplit/StringSplitV2:values:0Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
+text_vectorization_3/string_lookup_14/EqualEqual7text_vectorization_3/StringSplit/StringSplitV2:values:0-text_vectorization_3_string_lookup_14_equal_y*
T0*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/SelectV2SelectV2/text_vectorization_3/string_lookup_14/Equal:z:00text_vectorization_3_string_lookup_14_selectv2_tLtext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/IdentityIdentity7text_vectorization_3/string_lookup_14/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_3/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_3/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????,      ?
8text_vectorization_3/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_3/RaggedToTensor/Const:output:07text_vectorization_3/string_lookup_14/Identity:output:0:text_vectorization_3/RaggedToTensor/default_value:output:09text_vectorization_3/StringSplit/strided_slice_1:output:07text_vectorization_3/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_3/RaggedToTensor/RaggedTensorToTensor:result:0embedding_2_11791432*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791153?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_2/StatefulPartitionedCall:output:0conv1d_2_11791435conv1d_2_11791437*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791173?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791086?
flatten_2/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_11791186?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_11791325?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_11791443dense_4_11791445*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11791206?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_11791292?
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_5_11791449dense_5_11791451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_11791230w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv1d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCallD^text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2?
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_11792020

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:?????????Ȩ]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:?????????Ȩ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????Ȩ:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
?
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_11792067

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
P
__inference__creator_11792127
identity: ??StatefulPartitionedCall?
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
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6225046^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
O
__inference__creator_6167588
identity: ??StatefulPartitionedCall?
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
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6167584`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?	
f
G__inference_dropout_5_layer_call_and_return_conditional_losses_11792079

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
1
!__inference__initializer_11792133
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6225055G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
$__inference__traced_restore_11792422
file_prefix<
'assignvariableop_embedding_2_embeddings:???:
"assignvariableop_1_conv1d_2_kernel:??/
 assignvariableop_2_conv1d_2_bias:	?5
!assignvariableop_3_dense_4_kernel:
Ȩd-
assignvariableop_4_dense_4_bias:d3
!assignvariableop_5_dense_5_kernel:d-
assignvariableop_6_dense_5_bias:&
assignvariableop_7_adam_iter:	 (
assignvariableop_8_adam_beta_1: (
assignvariableop_9_adam_beta_2: (
assignvariableop_10_adam_decay: 0
&assignvariableop_11_adam_learning_rate: T
Jmutablehashtable_table_restore_lookuptableimportv2_statefulpartitionedcall: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: #
assignvariableop_14_total: #
assignvariableop_15_count: F
1assignvariableop_16_adam_embedding_2_embeddings_m:???B
*assignvariableop_17_adam_conv1d_2_kernel_m:??7
(assignvariableop_18_adam_conv1d_2_bias_m:	?=
)assignvariableop_19_adam_dense_4_kernel_m:
Ȩd5
'assignvariableop_20_adam_dense_4_bias_m:d;
)assignvariableop_21_adam_dense_5_kernel_m:d5
'assignvariableop_22_adam_dense_5_bias_m:F
1assignvariableop_23_adam_embedding_2_embeddings_v:???B
*assignvariableop_24_adam_conv1d_2_kernel_v:??7
(assignvariableop_25_adam_conv1d_2_bias_v:	?=
)assignvariableop_26_adam_dense_4_kernel_v:
Ȩd5
'assignvariableop_27_adam_dense_4_bias_v:d;
)assignvariableop_28_adam_dense_5_kernel_v:d5
'assignvariableop_29_adam_dense_5_bias_v:
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_2_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv1d_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv1d_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_4_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_4_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_5_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_5_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Jmutablehashtable_table_restore_lookuptableimportv2_statefulpartitionedcallRestoreV2:tensors:12RestoreV2:tensors:13*	
Tin0*

Tout0	**
_class 
loc:@StatefulPartitionedCall*
_output_shapes
 _
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp1assignvariableop_16_adam_embedding_2_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv1d_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv1d_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_4_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_4_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_5_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_5_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_embedding_2_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_conv1d_2_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv1d_2_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_4_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_4_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_5_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_5_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_class 
loc:@StatefulPartitionedCall
?
H
__inference__creator_6167580
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*8
shared_name)'table_1192406_load_4252035_load_6166387*
value_dtype0	Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
?
/__inference_sequential_2_layer_call_fn_11791262
input_3
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:???!
	unknown_4:??
	unknown_5:	?
	unknown_6:
Ȩd
	unknown_7:d
	unknown_8:d
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_dropout_5_layer_call_fn_11792057

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
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_11791217`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
/
__inference__destroyer_11792122
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6225035G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
.
__inference__destroyer_6166587
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6166582G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_11792005

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????H?  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:?????????ȨZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:?????????Ȩ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_4_layer_call_fn_11792015

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_11791325q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:?????????Ȩ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????Ȩ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
?
W
*__inference_restored_function_body_6225003
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference__creator_6166489^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
W
*__inference_restored_function_body_6166485
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference__creator_6166481`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
0
 __inference__initializer_6166803
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6166793G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_11791186

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????H?  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:?????????ȨZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:?????????Ȩ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
:
*__inference_restored_function_body_6166582
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__destroyer_6166578O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
f
G__inference_dropout_5_layer_call_and_return_conditional_losses_11791292

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
:
*__inference_restored_function_body_6225055
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__initializer_6166803O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
]
*__inference_restored_function_body_6225046
identity: ??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference__creator_6167588^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
.
__inference__destroyer_6166578
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_11791193

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:?????????Ȩ]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:?????????Ȩ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????Ȩ:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
ޖ
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791940

inputsT
Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handleU
Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value	1
-text_vectorization_3_string_lookup_14_equal_y4
0text_vectorization_3_string_lookup_14_selectv2_t	:
%embedding_2_embedding_lookup_11791886:???L
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:??7
(conv1d_2_biasadd_readvariableop_resource:	?:
&dense_4_matmul_readvariableop_resource:
Ȩd5
'dense_4_biasadd_readvariableop_resource:d8
&dense_5_matmul_readvariableop_resource:d5
'dense_5_biasadd_readvariableop_resource:
identity??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_2/embedding_lookup?Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2`
 text_vectorization_3/StringLowerStringLowerinputs*'
_output_shapes
:??????????
'text_vectorization_3/StaticRegexReplaceStaticRegexReplace)text_vectorization_3/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_3/SqueezeSqueeze0text_vectorization_3/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_3/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_3/StringSplit/StringSplitV2StringSplitV2%text_vectorization_3/Squeeze:output:0/text_vectorization_3/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_3/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_3/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_3/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_3/StringSplit/strided_sliceStridedSlice8text_vectorization_3/StringSplit/StringSplitV2:indices:0=text_vectorization_3/StringSplit/strided_slice/stack:output:0?text_vectorization_3/StringSplit/strided_slice/stack_1:output:0?text_vectorization_3/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_3/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_3/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_3/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_3/StringSplit/strided_slice_1StridedSlice6text_vectorization_3/StringSplit/StringSplitV2:shape:0?text_vectorization_3/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_3/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_3/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2LookupTableFindV2Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handle7text_vectorization_3/StringSplit/StringSplitV2:values:0Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
+text_vectorization_3/string_lookup_14/EqualEqual7text_vectorization_3/StringSplit/StringSplitV2:values:0-text_vectorization_3_string_lookup_14_equal_y*
T0*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/SelectV2SelectV2/text_vectorization_3/string_lookup_14/Equal:z:00text_vectorization_3_string_lookup_14_selectv2_tLtext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/IdentityIdentity7text_vectorization_3/string_lookup_14/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_3/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_3/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????,      ?
8text_vectorization_3/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_3/RaggedToTensor/Const:output:07text_vectorization_3/string_lookup_14/Identity:output:0:text_vectorization_3/RaggedToTensor/default_value:output:09text_vectorization_3/StringSplit/strided_slice_1:output:07text_vectorization_3/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding_2/embedding_lookupResourceGather%embedding_2_embedding_lookup_11791886Atext_vectorization_3/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*8
_class.
,*loc:@embedding_2/embedding_lookup/11791886*-
_output_shapes
:???????????*
dtype0?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_2/embedding_lookup/11791886*-
_output_shapes
:????????????
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_2/Conv1D/ExpandDims
ExpandDims0embedding_2/embedding_lookup/Identity_1:output:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
?
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims

??????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????h
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*-
_output_shapes
:???????????`
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
?
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????H?  ?
flatten_2/ReshapeReshape max_pooling1d_2/Squeeze:output:0flatten_2/Const:output:0*
T0*)
_output_shapes
:?????????Ȩ\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_4/dropout/MulMulflatten_2/Reshape:output:0 dropout_4/dropout/Const:output:0*
T0*)
_output_shapes
:?????????Ȩa
dropout_4/dropout/ShapeShapeflatten_2/Reshape:output:0*
T0*
_output_shapes
:?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*)
_output_shapes
:?????????Ȩ*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:?????????Ȩ?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:?????????Ȩ?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*)
_output_shapes
:?????????Ȩ?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
Ȩd*
dtype0?
dense_4/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_5/dropout/MulMuldense_4/Tanh:y:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:?????????dW
dropout_5/dropout/ShapeShapedense_4/Tanh:y:0*
T0*
_output_shapes
:?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_5/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_2/embedding_lookupD^text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2?
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791153

inputs	.
embedding_lookup_11791147:???
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_11791147inputs*
Tindices0	*,
_class"
 loc:@embedding_lookup/11791147*-
_output_shapes
:???????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/11791147*-
_output_shapes
:????????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????y
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:???????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_4_layer_call_and_return_conditional_losses_11792052

inputs2
matmul_readvariableop_resource:
Ȩd-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ȩd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????Ȩ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:?????????Ȩ
 
_user_specified_nameinputs
?
?
*__inference_restored_function_body_6167181
unknown
	unknown_0
	unknown_1	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__initializer_6167174`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??22
StatefulPartitionedCallStatefulPartitionedCall:"

_output_shapes

:??:"

_output_shapes

:??
?
:
*__inference_restored_function_body_6167162
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__destroyer_6167158O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791981

inputsC
+conv1d_expanddims_1_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims

?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????V
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:???????????g
IdentityIdentityRelu:activations:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?x
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791237

inputsT
Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handleU
Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value	1
-text_vectorization_3_string_lookup_14_equal_y4
0text_vectorization_3_string_lookup_14_selectv2_t	)
embedding_2_11791154:???)
conv1d_2_11791174:?? 
conv1d_2_11791176:	?$
dense_4_11791207:
Ȩd
dense_4_11791209:d"
dense_5_11791231:d
dense_5_11791233:
identity?? conv1d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2`
 text_vectorization_3/StringLowerStringLowerinputs*'
_output_shapes
:??????????
'text_vectorization_3/StaticRegexReplaceStaticRegexReplace)text_vectorization_3/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_3/SqueezeSqueeze0text_vectorization_3/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_3/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_3/StringSplit/StringSplitV2StringSplitV2%text_vectorization_3/Squeeze:output:0/text_vectorization_3/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_3/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_3/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_3/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_3/StringSplit/strided_sliceStridedSlice8text_vectorization_3/StringSplit/StringSplitV2:indices:0=text_vectorization_3/StringSplit/strided_slice/stack:output:0?text_vectorization_3/StringSplit/strided_slice/stack_1:output:0?text_vectorization_3/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_3/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_3/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_3/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_3/StringSplit/strided_slice_1StridedSlice6text_vectorization_3/StringSplit/StringSplitV2:shape:0?text_vectorization_3/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_3/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_3/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2LookupTableFindV2Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handle7text_vectorization_3/StringSplit/StringSplitV2:values:0Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
+text_vectorization_3/string_lookup_14/EqualEqual7text_vectorization_3/StringSplit/StringSplitV2:values:0-text_vectorization_3_string_lookup_14_equal_y*
T0*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/SelectV2SelectV2/text_vectorization_3/string_lookup_14/Equal:z:00text_vectorization_3_string_lookup_14_selectv2_tLtext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/IdentityIdentity7text_vectorization_3/string_lookup_14/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_3/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_3/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????,      ?
8text_vectorization_3/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_3/RaggedToTensor/Const:output:07text_vectorization_3/string_lookup_14/Identity:output:0:text_vectorization_3/RaggedToTensor/default_value:output:09text_vectorization_3/StringSplit/strided_slice_1:output:07text_vectorization_3/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_3/RaggedToTensor/RaggedTensorToTensor:result:0embedding_2_11791154*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791153?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_2/StatefulPartitionedCall:output:0conv1d_2_11791174conv1d_2_11791176*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791173?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791086?
flatten_2/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_11791186?
dropout_4/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_11791193?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_11791207dense_4_11791209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11791206?
dropout_5/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_11791217?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_5_11791231dense_5_11791233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_11791230w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv1d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCallD^text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2?
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
y
!__inference__initializer_11792116
unknown
	unknown_0
	unknown_1	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6225018G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??22
StatefulPartitionedCallStatefulPartitionedCall:"

_output_shapes

:??:"

_output_shapes

:??
?
-
__inference_<lambda>_11792184
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6225055J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
J
__inference__creator_11792104
identity??StatefulPartitionedCall?
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
 *0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_6225003^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
:
*__inference_restored_function_body_6225066
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__destroyer_6167167O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
:
*__inference_restored_function_body_6225035
identity?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__destroyer_6166587O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?{
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791655
input_3T
Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handleU
Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value	1
-text_vectorization_3_string_lookup_14_equal_y4
0text_vectorization_3_string_lookup_14_selectv2_t	)
embedding_2_11791632:???)
conv1d_2_11791635:?? 
conv1d_2_11791637:	?$
dense_4_11791643:
Ȩd
dense_4_11791645:d"
dense_5_11791649:d
dense_5_11791651:
identity?? conv1d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2a
 text_vectorization_3/StringLowerStringLowerinput_3*'
_output_shapes
:??????????
'text_vectorization_3/StaticRegexReplaceStaticRegexReplace)text_vectorization_3/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_3/SqueezeSqueeze0text_vectorization_3/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_3/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_3/StringSplit/StringSplitV2StringSplitV2%text_vectorization_3/Squeeze:output:0/text_vectorization_3/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_3/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_3/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_3/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_3/StringSplit/strided_sliceStridedSlice8text_vectorization_3/StringSplit/StringSplitV2:indices:0=text_vectorization_3/StringSplit/strided_slice/stack:output:0?text_vectorization_3/StringSplit/strided_slice/stack_1:output:0?text_vectorization_3/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_3/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_3/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_3/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_3/StringSplit/strided_slice_1StridedSlice6text_vectorization_3/StringSplit/StringSplitV2:shape:0?text_vectorization_3/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_3/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_3/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_3/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_3/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2LookupTableFindV2Ptext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_table_handle7text_vectorization_3/StringSplit/StringSplitV2:values:0Qtext_vectorization_3_string_lookup_14_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
+text_vectorization_3/string_lookup_14/EqualEqual7text_vectorization_3/StringSplit/StringSplitV2:values:0-text_vectorization_3_string_lookup_14_equal_y*
T0*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/SelectV2SelectV2/text_vectorization_3/string_lookup_14/Equal:z:00text_vectorization_3_string_lookup_14_selectv2_tLtext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
.text_vectorization_3/string_lookup_14/IdentityIdentity7text_vectorization_3/string_lookup_14/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_3/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_3/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????,      ?
8text_vectorization_3/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_3/RaggedToTensor/Const:output:07text_vectorization_3/string_lookup_14/Identity:output:0:text_vectorization_3/RaggedToTensor/default_value:output:09text_vectorization_3/StringSplit/strided_slice_1:output:07text_vectorization_3/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_3/RaggedToTensor/RaggedTensorToTensor:result:0embedding_2_11791632*
Tin
2	*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791153?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_2/StatefulPartitionedCall:output:0conv1d_2_11791635conv1d_2_11791637*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791173?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791086?
flatten_2/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_11791186?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????Ȩ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_11791325?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_11791643dense_4_11791645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11791206?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_11791292?
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_5_11791649dense_5_11791651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_11791230w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv1d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCallD^text_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):?????????: : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2?
Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2Ctext_vectorization_3/string_lookup_14/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
]
*__inference_restored_function_body_6167584
identity: ??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference__creator_6167580`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
?
?
+__inference_conv1d_2_layer_call_fn_11791965

inputs
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791173u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_30
serving_default_input_3:0?????????=
dense_52
StatefulPartitionedCall_3:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_sequential
u
	keras_api
_lookup_layer
#_self_saveable_object_factories
_adapt_function"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings
#_self_saveable_object_factories"
_tf_keras_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
#(_self_saveable_object_factories
 )_jit_compiled_convolution_op"
_tf_keras_layer
?
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
#0_self_saveable_object_factories"
_tf_keras_layer
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
#7_self_saveable_object_factories"
_tf_keras_layer
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_random_generator
#?_self_saveable_object_factories"
_tf_keras_layer
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
#H_self_saveable_object_factories"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O_random_generator
#P_self_saveable_object_factories"
_tf_keras_layer
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias
#Y_self_saveable_object_factories"
_tf_keras_layer
Q
1
&2
'3
F4
G5
W6
X7"
trackable_list_wrapper
Q
0
&1
'2
F3
G4
W5
X6"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
_trace_0
`trace_1
atrace_2
btrace_32?
/__inference_sequential_2_layer_call_fn_11791262
/__inference_sequential_2_layer_call_fn_11791717
/__inference_sequential_2_layer_call_fn_11791744
/__inference_sequential_2_layer_call_fn_11791507?
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
 z_trace_0z`trace_1zatrace_2zbtrace_3
?
ctrace_0
dtrace_1
etrace_2
ftrace_32?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791835
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791940
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791581
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791655?
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
 zctrace_0zdtrace_1zetrace_2zftrace_3
?B?
#__inference__wrapped_model_11791074input_3"?
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
?
giter

hbeta_1

ibeta_2
	jdecay
klearning_ratem?&m?'m?Fm?Gm?Wm?Xm?v?&v?'v?Fv?Gv?Wv?Xv?"
	optimizer
,
lserving_default"
signature_map
 "
trackable_dict_wrapper
"
_generic_user_object
q
m	keras_api
nlookup_table
otoken_counts
#p_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
?
qtrace_02?
__inference_adapt_step_6166561?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zqtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
wtrace_02?
.__inference_embedding_2_layer_call_fn_11791947?
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
 zwtrace_0
?
xtrace_02?
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791956?
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
 zxtrace_0
+:)???2embedding_2/embeddings
 "
trackable_dict_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?
~trace_02?
+__inference_conv1d_2_layer_call_fn_11791965?
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
 z~trace_0
?
trace_02?
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791981?
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
 ztrace_0
':%??2conv1d_2/kernel
:?2conv1d_2/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
2__inference_max_pooling1d_2_layer_call_fn_11791986?
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
 z?trace_0
?
?trace_02?
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791994?
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
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_flatten_2_layer_call_fn_11791999?
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
 z?trace_0
?
?trace_02?
G__inference_flatten_2_layer_call_and_return_conditional_losses_11792005?
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
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
,__inference_dropout_4_layer_call_fn_11792010
,__inference_dropout_4_layer_call_fn_11792015?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
G__inference_dropout_4_layer_call_and_return_conditional_losses_11792020
G__inference_dropout_4_layer_call_and_return_conditional_losses_11792032?
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
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_dense_4_layer_call_fn_11792041?
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
 z?trace_0
?
?trace_02?
E__inference_dense_4_layer_call_and_return_conditional_losses_11792052?
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
 z?trace_0
": 
Ȩd2dense_4/kernel
:d2dense_4/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
,__inference_dropout_5_layer_call_fn_11792057
,__inference_dropout_5_layer_call_fn_11792062?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
G__inference_dropout_5_layer_call_and_return_conditional_losses_11792067
G__inference_dropout_5_layer_call_and_return_conditional_losses_11792079?
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
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_dense_5_layer_call_fn_11792088?
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
 z?trace_0
?
?trace_02?
E__inference_dense_5_layer_call_and_return_conditional_losses_11792099?
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
 z?trace_0
 :d2dense_5/kernel
:2dense_5/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_sequential_2_layer_call_fn_11791262input_3"?
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
?B?
/__inference_sequential_2_layer_call_fn_11791717inputs"?
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
?B?
/__inference_sequential_2_layer_call_fn_11791744inputs"?
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
?B?
/__inference_sequential_2_layer_call_fn_11791507input_3"?
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
?B?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791835inputs"?
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
?B?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791940inputs"?
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
?B?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791581input_3"?
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
?B?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791655input_3"?
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
&__inference_signature_wrapper_11791690input_3"?
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
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
O
?_create_resource
?_initialize
?_destroy_resourceR Z

 ??
 "
trackable_dict_wrapper
?B?
__inference_adapt_step_6166561iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
.__inference_embedding_2_layer_call_fn_11791947inputs"?
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
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791956inputs"?
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
?B?
+__inference_conv1d_2_layer_call_fn_11791965inputs"?
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
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791981inputs"?
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
?B?
2__inference_max_pooling1d_2_layer_call_fn_11791986inputs"?
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
?B?
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791994inputs"?
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
?B?
,__inference_flatten_2_layer_call_fn_11791999inputs"?
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
G__inference_flatten_2_layer_call_and_return_conditional_losses_11792005inputs"?
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
?B?
,__inference_dropout_4_layer_call_fn_11792010inputs"?
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
?B?
,__inference_dropout_4_layer_call_fn_11792015inputs"?
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
?B?
G__inference_dropout_4_layer_call_and_return_conditional_losses_11792020inputs"?
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
?B?
G__inference_dropout_4_layer_call_and_return_conditional_losses_11792032inputs"?
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
?B?
*__inference_dense_4_layer_call_fn_11792041inputs"?
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
E__inference_dense_4_layer_call_and_return_conditional_losses_11792052inputs"?
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
?B?
,__inference_dropout_5_layer_call_fn_11792057inputs"?
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
?B?
,__inference_dropout_5_layer_call_fn_11792062inputs"?
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
?B?
G__inference_dropout_5_layer_call_and_return_conditional_losses_11792067inputs"?
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
?B?
G__inference_dropout_5_layer_call_and_return_conditional_losses_11792079inputs"?
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
?B?
*__inference_dense_5_layer_call_fn_11792088inputs"?
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
E__inference_dense_5_layer_call_and_return_conditional_losses_11792099inputs"?
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
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
?
?trace_02?
__inference__creator_11792104?
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
annotations? *? z?trace_0
?
?trace_02?
!__inference__initializer_11792116?
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
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_11792122?
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
annotations? *? z?trace_0
?
?trace_02?
__inference__creator_11792127?
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
annotations? *? z?trace_0
?
?trace_02?
!__inference__initializer_11792133?
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
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_11792139?
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
annotations? *? z?trace_0
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
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
?B?
__inference__creator_11792104"?
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
?B?
!__inference__initializer_11792116"?
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
?B?
__inference__destroyer_11792122"?
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
?B?
__inference__creator_11792127"?
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
?B?
!__inference__initializer_11792133"?
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
?B?
__inference__destroyer_11792139"?
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
0:.???2Adam/embedding_2/embeddings/m
,:*??2Adam/conv1d_2/kernel/m
!:?2Adam/conv1d_2/bias/m
':%
Ȩd2Adam/dense_4/kernel/m
:d2Adam/dense_4/bias/m
%:#d2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
0:.???2Adam/embedding_2/embeddings/v
,:*??2Adam/conv1d_2/kernel/v
!:?2Adam/conv1d_2/bias/v
':%
Ȩd2Adam/dense_4/kernel/v
:d2Adam/dense_4/bias/v
%:#d2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
?B?
__inference_save_fn_11792158checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_11792166restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant9
__inference__creator_11792104?

? 
? "? 9
__inference__creator_11792127?

? 
? "? ;
__inference__destroyer_11792122?

? 
? "? ;
__inference__destroyer_11792139?

? 
? "? D
!__inference__initializer_11792116n???

? 
? "? =
!__inference__initializer_11792133?

? 
? "? ?
#__inference__wrapped_model_11791074un???&'FGWX0?-
&?#
!?
input_3?????????
? "1?.
,
dense_5!?
dense_5?????????l
__inference_adapt_step_6166561Jo???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
F__inference_conv1d_2_layer_call_and_return_conditional_losses_11791981h&'5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
+__inference_conv1d_2_layer_call_fn_11791965[&'5?2
+?(
&?#
inputs???????????
? "?????????????
E__inference_dense_4_layer_call_and_return_conditional_losses_11792052^FG1?.
'?$
"?
inputs?????????Ȩ
? "%?"
?
0?????????d
? 
*__inference_dense_4_layer_call_fn_11792041QFG1?.
'?$
"?
inputs?????????Ȩ
? "??????????d?
E__inference_dense_5_layer_call_and_return_conditional_losses_11792099\WX/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? }
*__inference_dense_5_layer_call_fn_11792088OWX/?,
%?"
 ?
inputs?????????d
? "???????????
G__inference_dropout_4_layer_call_and_return_conditional_losses_11792020`5?2
+?(
"?
inputs?????????Ȩ
p 
? "'?$
?
0?????????Ȩ
? ?
G__inference_dropout_4_layer_call_and_return_conditional_losses_11792032`5?2
+?(
"?
inputs?????????Ȩ
p
? "'?$
?
0?????????Ȩ
? ?
,__inference_dropout_4_layer_call_fn_11792010S5?2
+?(
"?
inputs?????????Ȩ
p 
? "??????????Ȩ?
,__inference_dropout_4_layer_call_fn_11792015S5?2
+?(
"?
inputs?????????Ȩ
p
? "??????????Ȩ?
G__inference_dropout_5_layer_call_and_return_conditional_losses_11792067\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ?
G__inference_dropout_5_layer_call_and_return_conditional_losses_11792079\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? 
,__inference_dropout_5_layer_call_fn_11792057O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d
,__inference_dropout_5_layer_call_fn_11792062O3?0
)?&
 ?
inputs?????????d
p
? "??????????d?
I__inference_embedding_2_layer_call_and_return_conditional_losses_11791956b0?-
&?#
!?
inputs??????????	
? "+?(
!?
0???????????
? ?
.__inference_embedding_2_layer_call_fn_11791947U0?-
&?#
!?
inputs??????????	
? "?????????????
G__inference_flatten_2_layer_call_and_return_conditional_losses_11792005`5?2
+?(
&?#
inputs???????????
? "'?$
?
0?????????Ȩ
? ?
,__inference_flatten_2_layer_call_fn_11791999S5?2
+?(
&?#
inputs???????????
? "??????????Ȩ?
M__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_11791994?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
2__inference_max_pooling1d_2_layer_call_fn_11791986wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'???????????????????????????|
__inference_restore_fn_11792166YoK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_11792158?o&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791581qn???&'FGWX8?5
.?+
!?
input_3?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791655qn???&'FGWX8?5
.?+
!?
input_3?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791835pn???&'FGWX7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_11791940pn???&'FGWX7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_2_layer_call_fn_11791262dn???&'FGWX8?5
.?+
!?
input_3?????????
p 

 
? "???????????
/__inference_sequential_2_layer_call_fn_11791507dn???&'FGWX8?5
.?+
!?
input_3?????????
p

 
? "???????????
/__inference_sequential_2_layer_call_fn_11791717cn???&'FGWX7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
/__inference_sequential_2_layer_call_fn_11791744cn???&'FGWX7?4
-?*
 ?
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_11791690?n???&'FGWX;?8
? 
1?.
,
input_3!?
input_3?????????"1?.
,
dense_5!?
dense_5?????????