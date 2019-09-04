function sign(n){
	if(n >= 0 ){
		return 1;
	}else{
		return -1;
	}
}

class Perpectron {

    // Perpectron Constructor
	constructor(){
        this.weights = [0,0,0];
        const min = -1;
        const max = 1;

        for(let i = 0; i < this.weights.length; i++){
            this.weights[i] = Math.floor(Math.random() * (max - min + 1)) + min;
        }
        console.log('weights: ' + this.weights);  
    }

    guess(inputs){
        let sum = 0;
		for(let i = 0; i < this.weights.length; i++){
			sum += (inputs[i]*this.weights[i]);
		}
        console.log('sum: ' + sum);
        
        let output = sign(sum);
        console.log('output: ' + output);
		return output;
    }
    
    getX(){
        return this.weights;
    }
}
