%{
Active nematic hydrodynamics (ANHD)
Matt Golden 2022

This code was written in a hurry at the UMass Amherst Summer School on soft
solids and complex fluids. It evolves a set of active nematic hydrodynamic 
equations outlined in a paper from University of Nebraska on Exact Coherent 
Structures in active nematic channel flow. I stuck with their conventions, 
but I'd like to mention a few things.

1. This code is written for periodic boundary conditions. This allows us to
solve stiff differential equations (i.e. dissipative problems) implicitly
with the same numerical cost as an explicit solver thanks to the Fourier
transform. Nontrivial boundary 

2. In 2D, You do not need to solve for the 2D flow field <u,v>. The vorticity
\partial_y u - \partial_x v contains all the information of the flow,
except for the mean flow, which I assume vanishes. Since the active nematic
forcing is conservative, it is a constant of motion and will always vanish.

3. The elastic energy used in this paper is not the most general. There are
three possible "curvature" penalties in 2D. Two of them are the bend and
splay penalties, while the third does not come from any free energy. Reach
out to me if you care about this at some point. The code uses the standard
one constant approximation.

4. I added tumbling (flow alignment), which is neglected in the paper
because we have observed it to exist in the microtubule bundle case. 

5. I am not confident this code is bug-free

6. The integrator is second order in time and spectral in space. In other
words, You need less spatial resolution than you think you do, and more
time resolution than you think you do.

7. This integrator makes use of the scalar order parameter (magnitude of
the Q tensor), which I am not convinced is physical. I suppose there might
be a bijection with the density field in the correct formulation, but it is
unclear to me the scalar dynamics are anything close to what they ought to
be.
%}


%% Parameter Declaration
clear;

params.eta   = 1;   %dynamic viscosity
params.gamma = 1;   %rotational viscosity
params.K     = 0.1; %nematic elasticity
params.A     = 1;   %Bulk energy parameter
params.B     = 1;   %Bulk energy parameter
params.tumble= 1;   %Tumbling parameter (measured to be 1 for microtubule bundles by SPIDER)
params.alpha = 1;   %activity coefficient

N = 128; %Grid resolution
params.N = N; %also store it in params so it is easy to access

max_timestep = 1024;
dt = 0.1;
draw_every = 4;



%% Initial Conditions
[x,y] = meshgrid( (0:(N-1))/N*2*pi );

omega = sin(x).*sin(y) + cos(3*y - 0.1).*cos(2*x-0.7);

theta = atan2(y - pi + 1e-3,x - pi + 1e-2);
theta = theta + 0.5;
Q_11  = 2*cos(theta);
Q_12  = 2*sin(theta);

z = pack_z( omega, Q_11, Q_12 );

%% Integration

vidObj = VideoWriter('AN.avi');
open(vidObj);

q = zeros(3, max_timestep); 
%^projection coordinates. I was plotting these later to see how chaotic the
%flow looks.
for timesteps = 1:max_timestep
  z = mixed_rk2(z, dt, params );
  z = dealias_field(z, N);
  
  [q(:,timesteps), ~] = projection_coordinates(z, params);

  if mod(timesteps, draw_every) ~= 0
    continue;
  end  
  
  plot_z(z,N);
  axis tight
  drawnow
  
  
  % Write each frame to the file.
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
end
close(vidObj);

function z = pack_z( omega, Q_11, Q_12 )
  a = numel(omega);
  omega = reshape( omega, [a, 1] );
  Q_11  = reshape( Q_11,  [a, 1] );
  Q_12  = reshape( Q_12,  [a, 1] );

  z = [omega; Q_11; Q_12];
end

function  [omega, Q_11, Q_12] = unpack_z(z,N)
  omega = z(         1:N*N  );
  Q_11  = z(  N*N + (1:N*N) );
  Q_12  = z(2*N*N + (1:N*N) );

  omega = reshape( omega, [N, N] );
  Q_11  = reshape( Q_11,  [N, N] );
  Q_12  = reshape( Q_12,  [N, N] );
end

function f = rhs(z, params)
  %{
  Calculates the explicit part of the dynamics
  %}

  N = params.N;
  [omega, Q_11, Q_12] = unpack_z(z,N);

  k = 0:N-1;
  k(k>N/2) = k(k>N/2) - N;
  A = 1./(k.^2 + k'.^2);
  A(1,1) = 1;

  omega_fft = fft2( omega );
  psi_fft   = omega_fft.*A; %invert Laplacian in Fourier space
  U = real(ifft2( 1i*k'.*psi_fft     ));
  V = real(ifft2(-1i*    psi_fft.*k  ));

  A_11 = real(ifft2(  -k'.*psi_fft.*k   )); %(U_x - V_y)/2
  A_12 = real(ifft2(  psi_fft.*k.^2 - k'.^2.*psi_fft ))/2; %(U_y + V_x)/2

  domega_dx = real(ifft2(1i*omega_fft.*k));
  domega_dy = real(ifft2(1i*k'.*omega_fft));

  Q_11_fft = fft2(Q_11);
  Q_12_fft = fft2(Q_12);
  
  dQ_11_dx = real(ifft2(1i.*Q_11_fft.*k ));
  dQ_11_dy = real(ifft2(1i*k'.*Q_11_fft ));
  dQ_12_dx = real(ifft2(1i*Q_12_fft.*k  ));
  dQ_12_dy = real(ifft2(1i*k'.*Q_12_fft ));

  dQ_11_dxdy = real(ifft2( -k'.*Q_11_fft.*k));
  dQ_12_dxdx = real(ifft2( -Q_12_fft.*k.*k));
  dQ_12_dydy = real(ifft2( -k'.*k'.*Q_12_fft));

  active_vorticity_forcing = dQ_12_dxdx - dQ_12_dydy - 2*dQ_11_dxdy;

  domega_dt = -U.*domega_dx - V.*domega_dy - (params.alpha)*active_vorticity_forcing;


  %explicit parts of H
  q_sq = Q_11.^2 + Q_12.^2;
  H_11 = params.A*Q_11 - 2*params.B*Q_11.*q_sq;
  H_12 = params.A*Q_12 - 2*params.B*Q_12.*q_sq; 

  Q_11_normalized = Q_11./sqrt(Q_11.^2 + Q_12.^2)/2;
  Q_12_normalized = Q_12./sqrt(Q_11.^2 + Q_12.^2)/2;

  dQ_11_dt = -U.*dQ_11_dx - V.*dQ_11_dy + 2*omega.*Q_12 + params.gamma*H_11 + params.tumble*2*(Q_11.*A_11 + Q_12.*A_12).*(1 - 2*Q_11_normalized);
  dQ_12_dt = -U.*dQ_12_dx - V.*dQ_12_dy - 2*omega.*Q_11 + params.gamma*H_12 + params.tumble*2*(Q_11.*A_11 + Q_12.*A_12).*(0 - 2*Q_12_normalized);


  f = pack_z( domega_dt, dQ_11_dt, dQ_12_dt );
end


function zp = implicit_solve( z, dt, params )
  N = params.N; 
  [omega, Q_11, Q_12] = unpack_z( z, N );
  
  k = 0:N-1;
  k(k>N/2) = k(k>N/2) - N;
  
  omega_next     = real( ifft2( ...
                   fft2(omega)./(1 + dt*(params.eta)*(k.^2 + k'.^2) ) ...
                   ) );

  Q_11_next     = real( ifft2( ...
                   fft2(Q_11)./(1 + dt*(params.gamma)*params.K*(k.^2 + k'.^2) ) ...
                   ) );

  Q_12_next     = real( ifft2( ...
                   fft2(Q_12)./(1 + dt*(params.gamma)*params.K*(k.^2 + k'.^2) ) ...
                   ) );

  zp = pack_z( omega_next, Q_11_next, Q_12_next );
end

function zp = explicit_solve( z, dt, params )
  N = params.N; 
  [omega, Q_11, Q_12] = unpack_z( z, N );
  
  k = 0:N-1;
  k(k>N/2) = k(k>N/2) - N;
  
  omega_next     = real( ifft2( ...
                   fft2(omega).*(1 - dt*(params.eta)*(k.^2 + k'.^2) ) ...
                   ) );

  Q_11_next     = real( ifft2( ...
                   fft2(Q_11).*(1 - dt*(params.gamma)*params.K*(k.^2 + k'.^2) ) ...
                   ) );

  Q_12_next     = real( ifft2( ...
                   fft2(Q_12).*(1 - dt*(params.gamma)*params.K*(k.^2 + k'.^2) ) ...
                   ) );

  zp = pack_z( omega_next, Q_11_next, Q_12_next );
end



function zp = dealias_field(z, N)
  k = 0:N-1;
  k(k>N/2) = k(k>N/2) - N;
  
  mask = (k.^2 + k'.^2) < N/3;

  [omega, Q_11, Q_12] = unpack_z(z,N);

  omega2 = real(ifft2(mask.*(fft2(omega))));
  Q_11_2 = real(ifft2(mask.*(fft2(Q_11))));
  Q_12_2 = real(ifft2(mask.*(fft2(Q_12))));
  
  zp = pack_z(omega2, Q_11_2, Q_12_2 );
end


function zp = mixed_rk2(z, dt, params )
  %{
  PURPOSE:
  Solve a differential equation of the form \dot{x} = f(x) + gx, where
  f(x) is a nonlinear function to be evaluated explicitly, and g is a 
  dissipative matrix to handled implicitly for numerical stability

  This method is second order in time, which should be sufficient for
  playing around with dynamics
  %}
  f  = rhs(z, params ); %current velocity excluding dissipations
  z1 = z + f*dt; %explicit Euler for the nonlinear term
  z1 = implicit_solve( z1, dt, params ); %implicit Euler for dissipation
  f1 = rhs(z1, params); %caluclate the velocity at this guess z1
  
  zp = explicit_solve( z, dt/2, params );  %explicitly dissipate half a timestep
  zp = zp + dt*(f + f1)/2;                 %explciitly step forward in time averaging velocities
  zp = implicit_solve( zp, dt/2, params ); %finish with an implicit half-step
end

function plot_z(z,N)
  [omega, Q_11, Q_12] = unpack_z(z,N);

  tiledlayout(2,2);

  nexttile
  imagesc(omega);
  colorbar();
  axis square
  %caxis([-1 1]);
  title('vorticity')
  set(gca, 'ydir','normal');

  nexttile
  axis square
  imagesc( Q_11.^2 + Q_12.^2 );
  colorbar()
  title('Tr(Q^2)')
  axis square
  %caxis([0 1])
  set(gca, 'ydir','normal');
  
  nexttile
  theta = atan2(Q_12, Q_11)/2;
  n1 = cos(theta);
  n2 = sin(theta);
  [x,y] = meshgrid(1:N);

  d = 4;
  xs = x(1:d:end, 1:d:end);
  ys = y(1:d:end, 1:d:end);
  n1 = n1(1:d:end, 1:d:end);
  n2 = n2(1:d:end, 1:d:end);
  quiver( xs,ys,n1,n2,0.4, 'Color', 'black' );
  hold on
    quiver(xs, ys, -n1,-n2, 0.4,'Color','black')
  hold off
  axis square
  title('orientation')
  xlim([0 N]);
  ylim([0 N]);

  nexttile
  imagesc( Q_11./sqrt(Q_11.^2 + Q_12.^2 + 1e-6) );
  axis square
  caxis([-1 1])
  colorbar();
  title('normalized Q_{11}')
  set(gca, 'ydir','normal');
end

function [q, labels] = projection_coordinates(z, params)
  N = params.N;
  [omega, Q_11, Q_12] = unpack_z(z, N);
  
  q      = zeros(1,1); %projection coordinates
  labels = cell(1,1);  %string describing this coordinate
  a = 1;

  q(a)      = mean( 2*Q_11.^2 + 2*Q_12.^2, 'all' ); 
  labels{a} = "\langle Q_{ij} Q_{ij} \rangle";
  a = a+1;

  q(a)      = mean( omega.^2, 'all' );
  labels{a} = "\langle \omega^2 \rangle";
  a = a+1;

  q(a)      = std( omega.^2,0, 'all' );
  labels{a} = "std(omega^2)";
  a = a+1;
end