*정리: input x의 size(80,64,224,224)라 하면
             output
1.conv1   => (80,64,112,112)
2.MaxPool => (80,64,56,56)

3.Layer1  
-Block1   => (80,64,56,56)
-Block2   => (80,64,56,56)

3.Layer2  
-Block1   => (80,128,28,28)
-Block2   => (80,128,28,28)

3.Layer3  
-Block1   => (80,256,14,14)
-Block2   => (80,256,14,14)

3.Layer4  
-Block1   => (80,512,7,7)
-Block2   => (80,512,7,7)

*downsample이란 무엇인가?
basicblock에서 out += identity할 때 identity와 out의 size을 동일하게 만든다. 예를 들면 Layer2의 Block1을 보면 identity의 size는 (80,64,56,56)이고 
out의 size는 (80,128,28,28)이므로 out += identity이 안된다. 그걸 막기 위해서 downsample을 통해 
if self.downsample is not None:
      identity=self.downsample(out) #downsample 여부
identity와 out의 size를 동일하게 한다.
