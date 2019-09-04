class Point{
    constructor(){
        const min = 0;
        const max = 10;

        this.x = Math.floor(Math.random() * (max - min + 1)) + min;
        this.y = Math.floor(Math.random() * (max - min + 1)) + min;
        
        if(this.x > this.y){
            this.label = 'A'
        }
        else{
            this.label = 'B'
        }   
    }
}