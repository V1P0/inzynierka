import androidx.compose.desktop.ui.tooling.preview.Preview
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application

@Composable
@Preview
fun App() {

    MaterialTheme {
        var text by remember { mutableStateOf("") }
        Scaffold(
            bottomBar = {
                TextField(
                    value = text,
                    onValueChange = { text = it },
                    modifier = androidx.compose.ui.Modifier.fillMaxWidth(),
                    label = { Text("History") }
                )
            }
        ) {
            Board(text)
        }
    }
}

@Composable
fun Board(history:String){
    // draw a dot at the center of the screen
    val radius = with(LocalDensity.current) { 5.dp.toPx() }
    Canvas(modifier = androidx.compose.ui.Modifier.fillMaxWidth().fillMaxHeight()) {

        var centerX = size.width / 2
        var centerY = size.height / 2
        // moves = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
        val moves = listOf(
            Pair(0,1),
            Pair(1,1),
            Pair(1,0),
            Pair(1,-1),
            Pair(0,-1),
            Pair(-1,-1),
            Pair(-1,0),
            Pair(-1,1)
        )

        for(i in -3..3){
            for(j in -5..5){
                drawCircle(
                    color = Color.Black,
                    radius = 2f,
                    center = Offset(centerX - i * 20f, centerY - j * 20f)
                )
            }
        }

        val border = "2022200000000006660664666444444444422242"
        var topX = centerX
        var topY = centerY + 6*20

        for(char in border){
            val (updateX, updateY) = moves[char.toString().toInt()]
            drawLine(
                color = Color.Black,
                start = Offset(topX, topY),
                end = Offset(topX - updateX * 20f, topY - updateY * 20f),
                strokeWidth = 4f
            )
            topX -= updateX * 20
            topY -= updateY * 20
        }

        for(char in history){
            if(char == ' ') continue
            if(char == ';') break
            //check if char is a number
            if(char.toString().toIntOrNull() == null) continue
            if(char.toString().toInt() > 7) continue
            if(char.toString().toInt() < 0) continue
            val (updateX, updateY) = moves[char.toString().toInt()]
            drawCircle(
                color = Color.Blue,
                radius = radius,
                center = Offset(centerX - updateX * 20f, centerY - updateY * 20f)
            )
            drawLine(
                color = Color.Blue,
                start = Offset(centerX, centerY),
                end = Offset(centerX - updateX * 20f, centerY - updateY * 20f),
                strokeWidth = 4f
            )
            centerX -= updateX * 20
            centerY -= updateY * 20
        }
        drawCircle(
            color = Color.Red,
            radius = radius,
            center = Offset(centerX, centerY)
        )
    }
}

fun main() = application {
    Window(onCloseRequest = ::exitApplication) {
        App()
    }
}
