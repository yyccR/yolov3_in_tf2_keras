
```python
class VGG(nn.Module):
    def __init__(self, image_size=448):
        super(VGG, self).__init__()
        self.features = nn.Sequential(*layers)
        self.image_size = image_size
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1470),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.sigmoid(x)
        x = x.view(-1, 7, 7, 30)
        return x

```