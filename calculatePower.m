function [alpha_mu, beta_mu] = calculatePower(sig_, dc_)
    close all
    

    Fsig_ = fft(sig_) / length(sig_);
    aFsig_ = abs(Fsig_); 
   
    p_spectrum = aFsig_.^2; 
    p_spectrum = 2*p_spectrum(1:floor(length(Fsig_)/2));
    p_spectrum(1) = p_spectrum(1)/2;

    p_spectrum = 10*log10(p_spectrum / dc_);
    
    alpha_range = 8*5  : 12*5;
    beta_range  = 16*5 : 30*5;

    alpha = p_spectrum(alpha_range);
    beta  = p_spectrum(beta_range);

    alpha_mu = mean(alpha);
    beta_mu  = mean(beta);

end
    